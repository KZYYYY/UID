import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist

from loss.supcontrast import SupConLoss


def do_train_multi_view(cfg,
                        model,
                        train_loader_stage1,
                        optimizer,
                        scheduler,
                        local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    loss_I2T = AverageMeter()
    loss_T2I = AverageMeter()
    loss_INS = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train - extract image features
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_cas_features = []
    labels = []
    cam_labels = []
    with torch.no_grad():
        for n_iter, (img_cas, _, vid, target_cam) in enumerate(train_loader_stage1):
            img_cas = img_cas.to(device)
            target = vid.to(device)
            cam = target_cam.to(device)
            with amp.autocast(enabled=True):
                image_cas_feature = model(img_cas, target, get_image=True)
                cnt = 0
                for i, img_cas_feat in zip(target, image_cas_feature):
                    labels.append(i)
                    cam_labels.append(cam[cnt])
                    image_cas_features.append(img_cas_feat.cpu())
                    cnt += 1
        labels_list = torch.stack(labels, dim=0).cuda()  # N
        cam_labels_list = torch.stack(cam_labels, dim=0).cuda()
        image_cas_features_list = torch.stack(image_cas_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_cas_features

    # train - learn prompt

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_I2T.reset()
        loss_T2I.reset()
        loss_INS.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list]  # 获取ID
            target_cam = cam_labels_list[b_list]
            image_dct_features = image_cas_features_list[b_list]  # 获取ID相关图像特征
            with amp.autocast(enabled=True):
                mv_ins_feats, da_ins_cls_loss, ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss = model(
                    img_feats=image_dct_features, label=target, cam_label=target_cam, mvdc=True)
                text_features_view1, text_features_view2, text_features_view3 = model(label=target,
                                                                                      get_text=True)  # 获取ID相关文本特征
                # mv_ins_feats, da_ins_cls_loss, ins_mv_recon_loss, ins_mv_cls_loss = model(
                #     img_feats=image_dct_features, label=target, cam_label=target_cam, mvdc=True)
                # text_features_view1 = model(label=target, get_text=True)  # 获取ID相关文本特征
            # loss_i2t_view1 = xent(mv_ins_feats.half(), text_features_view1, target, target)
            # loss_t2i_view1 = xent(text_features_view1, mv_ins_feats.half(), target, target)

            loss_i2t_view1 = xent(mv_ins_feats[0].half(), text_features_view1, target, target)
            loss_t2i_view1 = xent(text_features_view1, mv_ins_feats[0].half(), target, target)

            loss_i2t_view2 = xent(mv_ins_feats[1].half(), text_features_view2, target, target)
            loss_t2i_view2 = xent(text_features_view2, mv_ins_feats[1].half(), target, target)

            loss_i2t_view3 = xent(mv_ins_feats[2].half(), text_features_view3, target, target)
            loss_t2i_view3 = xent(text_features_view3, mv_ins_feats[2].half(), target, target)

            loss_i2t = loss_i2t_view1 + loss_i2t_view2 + loss_i2t_view3
            loss_t2i = loss_t2i_view1 + loss_t2i_view2 + loss_t2i_view3

            # loss_i2t = loss_i2t_view1
            # loss_t2i = loss_t2i_view1

            loss_ins = (
                da_ins_cls_loss * cfg.MODEL.DA_INS_LOSS_WEIGHT
                + ins_mv_recon_loss * cfg.MODEL.MV_RECON_LOSS_WEIGHT
                + ins_mv_cls_loss * cfg.MODEL.CLS_INS_LOSS_WEIGHT
                + ins_mv_dis_loss * cfg.MODEL.INS_DIS_LOSS_WEIGHT
            )
            # loss_ins = da_ins_cls_loss * cfg.MODEL.DA_INS_LOSS_WEIGHT + ins_mv_recon_loss * cfg.MODEL.MV_RECON_LOSS_WEIGHT + ins_mv_cls_loss * cfg.MODEL.CLS_INS_LOSS_WEIGHT
            loss = (
                cfg.MODEL.STAGE1_I2T_LOSS_WEIGHT * loss_i2t
                + cfg.MODEL.STAGE1_T2I_LOSS_WEIGHT * loss_t2i
                + loss_ins
            )

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img_cas.shape[0])
            loss_I2T.update(loss_i2t.item(), img_cas.shape[0])
            loss_T2I.update(loss_t2i.item(), img_cas.shape[0])
            loss_INS.update(loss_ins.item(), img_cas.shape[0])
            loss_meter.update(loss.item(), img_cas.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}]ration[{}/{}] Loss: {:.3f}, Loss_i2t: {:.3f}, Loss_t2i: {:.3f}, Loss_ins: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (i + 1), len(train_loader_stage1),
                                loss_meter.avg, loss_I2T.avg, loss_T2I.avg, loss_INS.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0 and epoch > 40 or epoch == epochs:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_multi_view_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_multi_view_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
