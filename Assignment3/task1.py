# === Cell 1: Install dependencies ===
# !pip install -q torch torchvision pycocotools matplotlib transformers


# === Cell 2: Import libraries ===# =============================================================================
#  Imports & Device Setup
# =============================================================================
import os
import json
from tqdm import tqdm # type: ignore

import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore

from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from transformers.optimization import get_scheduler # type: ignore

from PIL import Image # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore
from pycocotools.coco import COCO # type: ignore
from pycocotools.cocoeval import COCOeval # type: ignore
from pathlib import Path
from collections import defaultdict

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check if GPU is available and print the device being used
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    
# =============================================================================
# Cell 3: Configuration
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 1. Paths & Output Directories
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_IMAGES_ROOT      = "/kaggle/input/a3-data/foggy_dataset_A3_train/foggy_dataset_A3_train"
TRAIN_ANNOTATIONS     = "/kaggle/input/a3-data/annotations_train.json"
VAL_IMAGES_ROOT        = "/kaggle/input/a3-data/foggy_dataset_A3_val/foggy_dataset_A3_val"
VAL_ANNOTATIONS       = "/kaggle/input/a3-data/annotations_val.json"

OUTPUT_DIR            = "outputs"
CHECKPOINT_DIR        = os.path.join(OUTPUT_DIR, "trained_checkpoints")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Training Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
NUM_EPOCHS            = 10 # Number of epochs to train
CHECKPOINT_FREQ       = 1  # every epoch 

# ──────────────────────────────────────────────────────────────────────────────
# 3. Evaluation Settings
# ──────────────────────────────────────────────────────────────────────────────
EVAL_THRESHOLDS       = [0.5]
NUM_IMAGES           = 5  # Number of images to visualize during evaluation

# ──────────────────────────────────────────────────────────────────────────────
# 4. Create Output Directories
# ──────────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
for thr in EVAL_THRESHOLDS:
    os.makedirs(os.path.join(OUTPUT_DIR, f"threshold_{thr}"), exist_ok=True)

    
    
# =============================================================================
# =============================================================================
# 3.  Dataset Definition
# =============================================================================
# Dataset class for training and evaluation
class FoggyDataset(Dataset):
    def __init__(self, img_rt: str, ann_path: str, proc: DeformableDetrImageProcessor, is_train: bool = True):
        self.img_rt = img_rt
        self.proc = proc
        self.is_train = is_train
        
        # Load annotations
        with open(ann_path) as f:
            self.annon = json.load(f)
        
        # Create mappings
        self.img_to_info = {img['id']: img for img in self.annon['images']}
        self.img_to_anns = defaultdict(list)
        
        for ann in self.annon['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        
        self.valid_img = list(self.img_to_anns.keys())
        
        # Category mappings
        self.category_id_to_category_name = {cat['id']: cat['name'] for cat in self.annon['categories']}
        self.category_name_to_category_id = {cat['name']: cat['id'] for cat in self.annon['categories']}
        
        
        self.cat_map = { 'person': 'person', 'car': 'car', 'train': 'train', 'rider': 'person', 'truck': 'truck', 'motorcycle': 'motorcycle', 'bicycle': 'bicycle', 'bus': 'bus' }
        
        # Create two-way mapping between dataset and model categories
        self.model_to_dataset = {}
        self.dataset_to_model = {}
        
        for cat_id, cat_name in self.category_id_to_category_name.items():
            if cat_name in self.cat_map:
                if cat_name == 'rider':continue
                model_name = self.cat_map[cat_name]
                if model_name in model.config.label2id:
                    model_id = model.config.label2id[model_name]
                    self.model_to_dataset[model_id] = cat_id
                    self.dataset_to_model[cat_id] = model_id
    def __len__(self):
        return len(self.valid_img)
    
    def __getitem__(self, idx):
        img_id = self.valid_img[idx]
        img_info = self.img_to_info[img_id]
        
        # Load image
        img_path = os.path.join(self.img_rt, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        annotations = self.img_to_anns[img_id]
        
        if self.is_train:
            coco_annotations = []
            for ann in annotations:
                if ann['category_id'] in self.dataset_to_model:
                    x, y, w, h = ann['bbox']
                    coco_annotations.append({
                        'id': ann['id'],
                        'image_id': img_id,
                        'category_id': self.dataset_to_model[ann['category_id']],
                        'bbox': [x, y, w, h],
                        'area': ann['area'],
                        'iscrowd': ann.get('iscrowd', 0)
                    })
            
            if not coco_annotations:
                coco_annotations.append({ 'id': -1, 'image_id': img_id, 'category_id': 0, 'bbox': [0, 0, 1, 1], 'area': 1, 'iscrowd': 0 })
            
            tgt = { 'image_id': img_id, 'annotations': coco_annotations }
            
            # Process for training
            inputs = self.proc(images=img, annotations=tgt, return_tensors="pt")
            
            # Remove batch dimension and ensure proper shapes
            result = {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'pixel_mask': inputs['pixel_mask'].squeeze(0),
                'labels': {
                    'class_labels': inputs['labels'][0]['class_labels'].view(-1),
                    'boxes': inputs['labels'][0]['boxes'].view(-1, 4),
                    'image_id': torch.tensor([img_id], dtype=torch.long),
                    'orig_size': inputs['labels'][0]['orig_size'],
                    'size': inputs['labels'][0]['size']
                }
            }
            
            return result
        else:
            # For evaluation/inference
            inputs = self.proc(images=img, return_tensors="pt")
            return inputs, annotations, img_id, img_path
    
def collate_fn(batch):
    """
    Collate function for DataLoader, handling both train and eval modes.

    Expects each item in `batch` to be:
      - if train mode: a dict with keys
            "pixel_values", "pixel_mask", "labels"
        where "labels" is itself a dict of tensors:
            "class_labels", "boxes", "image_id", "orig_size", "size"
      - if eval mode: a tuple (inputs, annotations, img_id, img_path)
    """
    # Detect train vs eval by inspecting first element
    first = batch[0]
    if isinstance(first, dict) and "labels" in first:
        # TRAIN mode
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_mask   = torch.stack([item["pixel_mask"]   for item in batch])

        labels = []
        for item in batch:
            lbl = item["labels"]
            labels.append({
                "class_labels": lbl["class_labels"],
                "boxes":        lbl["boxes"],
                "image_id":     lbl["image_id"],
                "orig_size":    lbl["orig_size"],
                "size":         lbl["size"],})

        return { "pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels, }
    else:
        return batch[0]
    
    
# =============================================================================
# 4.  Visualization Helper
# =============================================================================
# Visualization function
def visualize_after(image, outputs, dataset, threshold, img_path):
    plt.figure(figsize=(16, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    for score, label, box in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
        if label.item() in dataset.model_to_dataset:
            cat_id = dataset.model_to_dataset[label.item()]
            cat_name = dataset.category_id_to_name[cat_id]
            
            box = box.tolist()
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            
            rect = patches.Rectangle( (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none' )
            ax.add_patch(rect)
            ax.text(x, y, f"{cat_name}: {score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    save_path = os.path.join(OUTPUT_DIR, f"threshold_{threshold}", f"{Path(img_path).stem}_a3cv.jpg")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    
# =============================================================================



# =============================================================================
# 5.  Pretrained Model Evaluation
# =============================================================================
# === A. Qualitative Visualization of Pretrained Model ===
def visualize_pretrained_model(n_samples:int=5, threshold:float=0.5):
    import random
    print(f"\nVisualizing {n_samples} pretrained detections (threshold={threshold})")
    coco_gt = COCO(VAL_ANNOTATIONS)
    # pick random val images
    img_ids = random.sample(coco_gt.getImgIds(), n_samples)
    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(VAL_IMAGES_ROOT, info["file_name"])
        image = Image.open(img_path).convert("RGB")
        # inference + drawing
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad(): outputs = model(**inputs)
        res = processor.post_process_object_detection(
            outputs, threshold=threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        # plot
        plt.figure(figsize=(8,6)); ax=plt.gca()
        ax.imshow(image)
        for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"]):
            x0,y0,x1,y1 = bx.cpu().tolist()
            ax.add_patch(patches.Rectangle((x0,y0),x1-x0,y1-y0,
                                           fill=False, edgecolor="red", lw=2))
            ax.text(x0,y0,f"{model.config.id2label[lb.item()]}:{sc:.2f}",
                    bbox={"facecolor":"yellow","alpha":0.5})
        ax.axis("off")
        plt.show()
        
        
# Evaluation function
def evaluate_model(model, dataset, processor, thresholds=[0.5], visualize=True):
    results = {}
    coco_gt = COCO(VAL_ANNOTATIONS)
    
    for threshold in thresholds:
        print(f"\Calculating at threshold {threshold}")
        coco_dt, ann_id = [], 1
        
        model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
        
        for batch in tqdm(dataloader, desc=f"Processing (threshold={threshold})"):
            if isinstance(batch, tuple) and len(batch) == 4: inputs, a, img_id, img_path = batch
            else:
                inputs = batch['pixel_values']
                img_id = batch['image_id']
                img_path = batch['img_path']
            
            # Move inputs to device
            pxl_val = inputs['pixel_values'].to(device)
            pxl_mask = inputs['pixel_mask'].to(device)
        
            with torch.no_grad():
                outputs = model(pixel_values=pxl_val, pixel_mask=pxl_mask)

            image = Image.open(img_path)
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            processed_outputs = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
            
            for score, label, box in zip(processed_outputs["scores"], processed_outputs["labels"], processed_outputs["boxes"]):
                if label.item() in dataset.model_to_dataset:
                    min_x, min_y, max_x, max_y = box.tolist()
                    w = max_x - min_x
                    h = max_y - min_y
                    
                    coco_dt.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": dataset.model_to_dataset[label.item()],
                        "bbox": [round(x, 2) for x in [min_x, min_y, w, h]],
                        "score": round(score.item(), 3)
                    })
                    ann_id += 1
            
            if visualize and len(coco_dt) > 0 and ann_id <= NUM_IMAGES * 10:
                visualize_after(image, processed_outputs, dataset, threshold, img_path)
                    
        
        predictions_file = os.path.join(OUTPUT_DIR, f"predictions_{threshold}.json")
        with open(predictions_file, 'w') as f:
            json.dump(coco_dt, f)
        
        coco_dt_list = coco_dt

        if not coco_dt_list:
            print(f"There is No predictions for threshold {threshold}; setting mAP to 0.0")
            results[threshold] = {
                'mAP':       0.0,
                'mAP_50':    0.0,
                'mAP_75':    0.0,
                'mAP_small': 0.0,
                'mAP_medium':0.0,
                'mAP_large': 0.0
            }
        else:
            coco_res = coco_gt.loadRes(coco_dt_list)
            coco_eval = COCOeval(coco_gt, coco_res, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            print(f"Results at the threshold {threshold}:")
            coco_eval.summarize()

            results[threshold] = {
                'mAP': coco_eval.stats[0],
                'mAP_50': coco_eval.stats[1],
                'mAP_75': coco_eval.stats[2],
                'mAP_small': coco_eval.stats[3],
                'mAP_medium': coco_eval.stats[4],
                'mAP_large': coco_eval.stats[5],
            }
    
    return results


# =============================================================================
# 6. Training Loop
# =============================================================================
# Training function
def train_model(model, train_dataset, val_dataset, processor):
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)
    
    best_mAP = 0.0
    train_losses, val_metrics = [], []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nepoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            pxl_val = batch["pixel_values"].to(device)
            pxl_mask = batch["pixel_mask"].to(device)
            
            lb = []
            for label_dict in batch["labels"]:
                device_labels = {
                    "class_labels": label_dict["class_labels"].to(device),
                    "boxes": label_dict["boxes"].to(device),
                    "image_id": label_dict["image_id"].to(device),
                    "orig_size": label_dict["orig_size"].to(device),
                    "size": label_dict["size"].to(device)
                }
                lb.append(device_labels)
            
            # Forward pass
            outputs = model(pixel_values=pxl_val, pixel_mask=pxl_mask, labels=lb)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            
            optimizer.step()
            lr_scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Evaluation phase
        if (epoch + 1) % 1 == 0:
            current_metrics = evaluate_model(model, val_dataset, processor, thresholds=EVAL_THRESHOLDS, visualize=(epoch == 0))
            val_metrics.append(current_metrics)
            
            current_mAP = current_metrics[0.5]['mAP']
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
                print(f"best model saved with mAP: {best_mAP:.4f}")
            
            if (epoch + 1) % CHECKPOINT_FREQ == 0:
                checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints", f"epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1}")
    
    return train_losses, val_metrics

def training_mode(mode):
    if mode == "encoder_only":
        print("Encoder-only training")
        # Freeze decoder and other components
        for param in model.model.decoder.parameters():
            param.requires_grad = False
        for param in model.class_embed.parameters():
            param.requires_grad = False
        for param in model.bbox_embed.parameters():
            param.requires_grad = False
        # Freeze the backbone
        for param in model.model.backbone.parameters():
            param.requires_grad = False
            
    elif mode == "decoder_only":
        print("Decoder-only training")
        # Freeze encoder and other components
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    else:
        print("Training the full model")

# Main function to run the training and evaluation
# === Main Execution ===
if __name__ == "__main__":
    # task1.sh <subtask> <experiment> <mode> <input dir>
    # Load the pretrained model and processor
    print("\nLoading pretrained model and processor...")
    processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

    model.to(device)
    # === A. Pretrained Model Evaluation (qualitative baseline) ===
    print("\n=== Baseline Pretrained Qualitative Eval ===")
    visualize_pretrained_model(n_samples=10, threshold=0.5)
    
    # === B. Standalone Pretrained Evaluation (quantitative baseline) ===
    print("\n=== Baseline Pretrained Quantitative Eval ===")
    evaluate_model(model, FoggyDataset(VAL_IMAGES_ROOT, VAL_ANNOTATIONS, processor, is_train=False), processor, thresholds=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95], visualize=False)
    
    # ===== Select training mode =====
    mode = "full"  # Options: "encoder_only", "decoder_only", "full"
    training_mode(mode)
    
    # Initialize datasets
    print("\nInitializing datasets...")
    train_dataset = FoggyDataset(TRAIN_IMAGES_ROOT, TRAIN_ANNOTATIONS, processor, is_train=True)
    val_dataset = FoggyDataset(VAL_IMAGES_ROOT, VAL_ANNOTATIONS, processor, is_train=False)
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_metrics = train_model(model, train_dataset, val_dataset, processor)
    
    # Save training results
    res = { 'train_losses': train_losses, 'val_metrics': val_metrics, }
    
    final_res_file = os.path.join(OUTPUT_DIR, "training_res_full.json")
    with open(final_res_file, 'w') as f: json.dump(res, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {final_res_file}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'train_loss_full.png'))
    plt.close()
    
    print("\nFinal Evaluation Results:")
    for epoch, metrics in enumerate(val_metrics):
        print(f"\nEpoch {epoch + 1}:")
        for threshold, values in metrics.items():
            print(f"\t Threshold {threshold}:")
            print(f"\t mAP @ [0.5:0.95]: {values['mAP']:.4f}")
            print(f"\t mAP @ 0.5: {values['mAP_50']:.4f}")
            
    print("\nTraining and evaluation completed!")