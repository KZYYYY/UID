import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_stage1(cfg,
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

    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train - extract image features
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    cam_labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image=True)
                cnt = 0
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    cam_labels.append(target_cam[cnt])
                    image_features.append(img_feat.cpu())
                    cnt += 1
        labels_list = torch.stack(labels, dim=0).cuda()  # N
        cams_list = torch.stack(cam_labels, dim=0).cuda()
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features, cam_labels

    # train - learn prompt
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_I2T.reset()
        loss_T2I.reset()
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
            target_cam = cams_list[b_list]
            image_features = image_features_list[b_list]  # 获取ID相关图像特征

            with amp.autocast(enabled=True):
                mv_ins_img_feat, da_ins_cls_loss, ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss = model(
                    img_feats=image_features, cam_label=target_cam, mvdc=True)
                text_features_view1, text_features_view2, text_features_view3 = model(label=target,
                                                                                      get_text=True)  # 获取ID相关文本特征

            loss_i2t_view1 = xent(mv_ins_img_feat[0].half(), text_features_view1, target, target)
            loss_t2i_view1 = xent(text_features_view1, mv_ins_img_feat[0].half(), target, target)
            loss_i2t_view2 = xent(mv_ins_img_feat[1].half(), text_features_view2, target, target)
            loss_t2i_view2 = xent(text_features_view2, mv_ins_img_feat[1].half(), target, target)
            loss_i2t_view3 = xent(mv_ins_img_feat[2].half(), text_features_view3, target, target)
            loss_t2i_view3 = xent(text_features_view1, mv_ins_img_feat[2].half(), target, target)

            loss_i2t = loss_i2t_view1 + loss_i2t_view2 + loss_i2t_view3
            loss_t2i = loss_t2i_view1 + loss_t2i_view2 + loss_t2i_view3

            loss_ins = 0.1 * da_ins_cls_loss + 10 * ins_mv_recon_loss + 0.2 * ins_mv_cls_loss + ins_mv_dis_loss

            loss = loss_i2t + loss_t2i + loss_ins

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            loss_I2T.update(loss_i2t.item(), img.shape[0])
            loss_T2I.update(loss_t2i.item(), img.shape[0])
            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss_i2t: {:.3f}, Loss_t2i: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch, (i + 1), len(train_loader_stage1),
                                loss_meter.avg, loss_I2T.avg, loss_T2I.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0 and epoch > 40 or epoch == epochs:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
