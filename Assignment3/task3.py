# Cell 1: Install dependencies
# !pip install -q ultralytics pycocotools

# Cell 2: Imports
import os, json, shutil, yaml #type:ignore
import numpy as np #type:ignore
import torch #type:ignore
from ultralytics import YOLO #type:ignore
from tqdm import tqdm #type:ignore
from pycocotools.coco import COCO #type:ignore
from pycocotools.cocoeval import COCOeval #type:ignore
import cv2 #type:ignore
import matplotlib.pyplot as plt #type:ignore

# Cell 3: COCO → YOLO label conversion
def coco_to_yolo(coco_json, img_dir, lbl_dir, coco_cat_ids):
    coco = COCO(coco_json)
    imgs = {img['id']: img for img in coco.loadImgs(coco.getImgIds())}
    anns = coco.loadAnns(coco.getAnnIds())
    by_img = {}
    for a in anns:
        by_img.setdefault(a['image_id'], []).append(a)

    for img_id, info in tqdm(imgs.items(), desc=f"Converting {os.path.basename(coco_json)}"):
        # preserve sub‐folders
        relpath = info['file_name'].replace('\\','/')
        sub, fname = os.path.split(relpath)
        stem = os.path.splitext(fname)[0]
        out_dir = os.path.join(lbl_dir, sub)
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, f"{stem}.txt")

        w, h = info['width'], info['height']
        with open(txt_path, 'w') as f:
            for a in by_img.get(img_id, []):
                cid = a['category_id']
                if cid not in coco_cat_ids: 
                    continue
                cls = coco_cat_ids.index(cid)
                x, y, bw, bh = a['bbox']
                x_c = (x + bw/2) / w
                y_c = (y + bh/2) / h
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {bw/w:.6f} {bh/h:.6f}\n")

# Cell 4: Copy images into train/val image folders
def copy_images(src_root, dst_root):
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(('.jpg','jpeg','png','bmp','tif','tiff')):
                rel = os.path.relpath(root, src_root)
                out_dir = os.path.join(dst_root, rel)
                os.makedirs(out_dir, exist_ok=True)
                shutil.copy(os.path.join(root, fn), os.path.join(out_dir, fn))

# Cell 5: Write data_config.yaml
def write_data_config(train_img_dir, val_img_dir, nc, names, out_path):
    cfg = {'train': train_img_dir, 'val': val_img_dir, 'nc': nc, 'names': names}
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f)
    print(f"Written YOLO data config → {out_path}")

# Cell 6: Zero-shot inference, evaluation & visualization
class FoggyDataset:
    def __init__(self, image_root, annotations_path):
        self.image_root = image_root
        self.coco = COCO(annotations_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.catmap = {1:'person',2:'car',3:'train',4:'rider',
                       5:'truck',6:'motorcycle',7:'bicycle',8:'bus'}
    def __len__(self): return len(self.image_ids)
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        fn = self.coco.imgs[img_id]['file_name']
        return {'image_id':img_id, 'image_path':os.path.join(self.image_root,fn)}

def generate_zero_shot_predictions(model, ds, out_json):
    results, aid = [], 1
    y2c = {0:1,1:3,2:2,3:6,5:8,7:5}
    for s in tqdm(ds, desc="Zero-shot"):
        preds = model.predict(s['image_path'], conf=0.01, iou=0.5, verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()
        cls   = preds.boxes.cls.cpu().numpy().astype(int)
        for b,cid,cf in zip(boxes, cls, confs):
            if cid not in y2c: continue
            x1,y1,x2,y2 = b; w,h = x2-x1,y2-y1
            results.append({'id':aid,'image_id':s['image_id'],
                            'category_id':y2c[cid],
                            'bbox':[float(x1),float(y1),float(w),float(h)],
                            'score':float(cf)})
            aid += 1
    with open(out_json,'w') as f: json.dump(results,f,indent=2)
    return out_json

def evaluate_predictions(gt, pred):
    coco_gt = COCO(gt)
    preds   = json.load(open(pred))
    coco_dt = coco_gt.loadRes(preds)
    ce = COCOeval(coco_gt, coco_dt,'bbox')
    ce.evaluate(); ce.accumulate(); ce.summarize()
    return ce.stats

def visualize_predictions(model, ds, num=3):
    idxs = np.random.choice(len(ds), num, replace=False)
    fig,axes = plt.subplots(num,1,figsize=(12,5*num))
    if num==1: axes=[axes]
    y2c = {0:1,1:7,2:2,3:6,5:8,7:5}
    colors = {i:tuple(np.random.randint(0,255,3).tolist()) for i in ds.catmap}
    for ax,i in zip(axes,idxs):
        s=ds[i]
        img=cv2.cvtColor(cv2.imread(s['image_path']),cv2.COLOR_BGR2RGB)
        gt=img.copy()
        for a in ds.coco.loadAnns(ds.coco.getAnnIds(imgIds=s['image_id'])):
            if a.get('iscrowd'): continue
            x,y,w,h=map(int,a['bbox'])
            c=colors[a['category_id']]
            cv2.rectangle(gt,(x,y),(x+w,y+h),c,2)
        pr=img.copy()
        preds=model.predict(s['image_path'],conf=0.25,iou=0.0,verbose=False)[0]
        for b,cf,cid in zip(preds.boxes.xyxy.cpu().numpy(),
                            preds.boxes.conf.cpu().numpy(),
                            preds.boxes.cls.cpu().numpy().astype(int)):
            if cid not in y2c: continue
            c=colors[y2c[cid]]; x1,y1,x2,y2=map(int,b)
            cv2.rectangle(pr,(x1,y1),(x2,y2),c,2)
        ax.imshow(np.hstack((gt,pr))); ax.axis('off')
    plt.tight_layout(); plt.savefig('t3_vis.png'); plt.close()
    print("Saved visualization→ t3_vis.png")

# Cell 7: Main
def main():
    # Paths
    TRAIN_IMG = "/kaggle/input/a3-data/foggy_dataset_A3_train/foggy_dataset_A3_train"
    VAL_IMG   = "/kaggle/input/a3-data/foggy_dataset_A3_val/foggy_dataset_A3_val"
    TRAIN_JS  = "/kaggle/input/a3-data/annotations_train.json"
    VAL_JS    = "/kaggle/input/a3-data/annotations_val.json"
    OUT       = "/kaggle/working/task3_yolo"
    os.makedirs(OUT, exist_ok=True)

    # 1) Convert labels
    cats = [1,2,3,4,5,6,7,8]
    train_lbl = os.path.join(OUT,"train","labels")
    val_lbl   = os.path.join(OUT,"val","labels")
    coco_to_yolo(TRAIN_JS, TRAIN_IMG, train_lbl, cats)
    coco_to_yolo(  VAL_JS,   VAL_IMG, val_lbl,   cats)

    # 2) Copy images
    train_img_out = os.path.join(OUT,"train","images")
    val_img_out   = os.path.join(OUT,"val","images")
    copy_images(TRAIN_IMG, train_img_out)
    copy_images(  VAL_IMG,   val_img_out)

    # 3) Write data_config.yaml
    names = ['person','car','train','rider','truck','motorcycle','bicycle','bus']
    dc = os.path.join(OUT,"data_config.yaml")
    write_data_config(
        train_img_dir=train_img_out, 
        val_img_dir=  val_img_out,
        nc=len(names), 
        names=names, 
        out_path=dc
    )

    # 4) Zero‐shot → eval → vis
    model = YOLO('yolov8x.pt')
    zs = os.path.join(OUT,"zero_shot.json")
    generate_zero_shot_predictions(model, FoggyDataset(VAL_IMG,VAL_JS), zs)
    print("\nZero‐Shot Metrics:")
    evaluate_predictions(VAL_JS, zs)
    visualize_predictions(model, FoggyDataset(VAL_IMG,VAL_JS), num=3)

    # 5) Fine‐tune
    print("\n▶️ Fine‐tuning YOLOv8x…")
    model.train(
      data=dc,
      epochs=10,
      batch=16,
      imgsz=640,
      project=OUT,
      name='yolov8x_ft',
      exist_ok=True
    )

    # 6) Re‐evaluate & vis
    best = YOLO(os.path.join(OUT,'yolov8x_ft','weights','best.pt'))
    ftf = os.path.join(OUT,"finetuned.json")
    generate_zero_shot_predictions(best, FoggyDataset(VAL_IMG,VAL_JS), ftf)
    print("\nFine‐Tuned Metrics:")
    evaluate_predictions(VAL_JS, ftf)
    visualize_predictions(best, FoggyDataset(VAL_IMG,VAL_JS), num=3)

if __name__=="__main__":
    main()