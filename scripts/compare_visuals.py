import os
from PIL import Image

def make_side_by_side(img1_path, img2_path, out_path):
    a = Image.open(img1_path).convert('RGB')
    b = Image.open(img2_path).convert('RGB')
    # resize b to match a height
    if a.size != b.size:
        b = b.resize(a.size, Image.BILINEAR)
    w, h = a.size
    out = Image.new('RGB', (w * 2, h))
    out.paste(a, (0, 0))
    out.paste(b, (w, 0))
    out.save(out_path)


def compare_pca(trained_dir, orig_dir, out_dir):
    for split in ['query', 'gallery']:
        tr_dir = os.path.join(trained_dir, f'vit_patch_pca_opri_{split}_20ids')
        or_dir = os.path.join(orig_dir, f'vit_patch_pca_opri_{split}_20ids')
        if not os.path.isdir(tr_dir) or not os.path.isdir(or_dir):
            print('Missing dir:', tr_dir, or_dir)
            continue
        save_dir = os.path.join(out_dir, f'compare_patch_pca_{split}_20ids')
        os.makedirs(save_dir, exist_ok=True)
        tr_files = sorted([f for f in os.listdir(tr_dir) if f.endswith('.png')])
        for fname in tr_files:
            tr_path = os.path.join(tr_dir, fname)
            or_path = os.path.join(or_dir, fname)
            if not os.path.isfile(or_path):
                continue
            out_path = os.path.join(save_dir, fname)
            make_side_by_side(tr_path, or_path, out_path)
        print('Saved comparisons to', save_dir)


def compare_tsne(trained_dir, orig_dir, out_dir):
    # assume tsne filenames like vit_cls_tsne_opri_both_20ids.png
    pairs = [
        ('vit_cls_tsne_opri_both_20ids.png', 'vit_cls_tsne_opri_both_20ids.png'),
        ('vit_cls_tsne_opri_query_20ids.png', 'vit_cls_tsne_opri_query_20ids.png'),
        ('vit_cls_tsne_opri_gallery_20ids.png', 'vit_cls_tsne_opri_gallery_20ids.png'),
    ]
    os.makedirs(out_dir, exist_ok=True)
    for tr_name, or_name in pairs:
        tr_path = os.path.join(trained_dir, tr_name)
        or_path = os.path.join(orig_dir, or_name)
        if not os.path.isfile(tr_path) or not os.path.isfile(or_path):
            continue
        out_path = os.path.join(out_dir, 'compare_' + tr_name)
        make_side_by_side(tr_path, or_path, out_path)
        print('Saved TSNE comparison to', out_path)


if __name__ == '__main__':
    base_tr = 'results/visualization'
    base_or = 'results/visualization/clip_orig2'
    out = 'results/visualization/compare'
    os.makedirs(out, exist_ok=True)
    compare_pca(base_tr, base_or, out)
    compare_tsne(base_tr, base_or, out)
    print('Done')
