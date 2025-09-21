# === Cell 1: Install Dependencies ===
# !pip install -q transformers==4.51.1 pycocotools matplotlib


# === Cell 2: Imports & Device Setup ===
import os, json, random
from pathlib import Path
from typing import List, Dict

import torch # type: ignore
from torch import nn # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from tqdm.auto import tqdm # type: ignore

from PIL import Image # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    get_scheduler,
    BertTokenizer, BertConfig, BertModel,
)

from pycocotools.coco import COCO # type: ignore
from pycocotools.cocoeval import COCOeval # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# === Cell 3: Paths & Hyperparameters ===
TRAIN_ROOT     = "/kaggle/input/a3-data/foggy_dataset_A3_train/foggy_dataset_A3_train"
TRAIN_ANN      = "/kaggle/input/a3-data/annotations_train.json"
VAL_ROOT       = "/kaggle/input/a3-data/foggy_dataset_A3_val/foggy_dataset_A3_val"
VAL_ANN        = "/kaggle/input/a3-data/annotations_val.json"

OUTPUT_DIR     = "gdino_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE     = 2
NUM_EPOCHS     = 5   # for prompt tuning only
LR_PROMPT      = 1e-5
NUM_PROMPT_TOK = 10  # length of learned prefix


# === Cell 4: Inference-Only Dataset ===
class CocoInferenceDataset(Dataset):
    def __init__(self, img_root: str, ann_file: str, processor: AutoProcessor):
        self.img_root  = img_root
        self.processor = processor
        self.coco      = COCO(ann_file)
        self.ids       = [i for i in self.coco.getImgIds() 
                         if len(self.coco.getAnnIds(imgIds=i)) > 0]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        path   = os.path.join(self.img_root, info["file_name"])
        img    = Image.open(path).convert("RGB")
        return img_id, path, img

def collate_inference(batch):
    return batch  # list of (img_id, path, img)


# === Cell 5: Inference & COCO Eval Utilities ===
def run_zero_shot(
    model: AutoModelForZeroShotObjectDetection,
    processor: AutoProcessor,
    dataset: CocoInferenceDataset,
    prompt: str,
    box_thresh: float = 0.4,
    text_thresh: float = 0.25,
    batch_size: int = 1,
    out_json: str = "gdino_zs_results.json",
):
    coco_gt = COCO(VAL_ANN)
    dets: List[Dict] = []
    ann_id = 1

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_inference)
    model.eval()
    for batch in tqdm(loader, desc="Zero-Shot Inference"):
        for img_id, path, img in batch:
            inputs = processor(images=img, text=[prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_thresh,
                text_threshold=text_thresh,
                target_sizes=[img.size[::-1]],
            )[0]

            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if isinstance(label, torch.Tensor):
                    cat_id = int(label.item())
                else:
                    names = coco_gt.getCatIds(catNms=[label])
                    cat_id = names[0] if names else -1
                x0, y0, x1, y1 = box.tolist()
                dets.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [x0, y0, x1 - x0, y1 - y0],
                    "score":       float(score.item()),
                })
                ann_id += 1

    with open(out_json, "w") as f:
        json.dump(dets, f)
    print(f"Saved zero-shot results to {out_json}")
    return dets

def coco_eval_from_list(coco_gt: COCO, dets: List[Dict]):
    coco_dt  = coco_gt.loadRes(dets)
    coco_eval= COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval.stats



# === Cell 6: Baseline Zero-Shot Eval ===
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model     = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(device)

val_ds   = CocoInferenceDataset(VAL_ROOT, VAL_ANN, processor)
baseline_prompt = "person. car. bus. train. truck. motorcycle. bicycle."

print("\n>>> Running baseline zero-shot inference …")
zs_dets = run_zero_shot(
    model, processor, val_ds,
    prompt=baseline_prompt,
    box_thresh=0.4, text_thresh=0.25,
    batch_size=4,
    out_json=os.path.join(OUTPUT_DIR, "zs_baseline.json")
)

print("\n>>> Quantitative Zero-Shot Eval:")
coco_gt = COCO(VAL_ANN)
_ = coco_eval_from_list(coco_gt, zs_dets)



# === Cell 7: Prompt‐Tuning Setup ===
from transformers import BertTokenizer

# 1) extend the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
new_tokens = [f"[PROMPT{i}]" for i in range(NUM_PROMPT_TOK)]
tokenizer.add_tokens(new_tokens)

# 2) resize the text‐backbone’s embeddings (the BERT under model.model.text_backbone)
try:
    # HF helper (will call get_input_embeddings() / set_input_embeddings() under the hood)
    model.model.text_backbone.resize_token_embeddings(len(tokenizer))
except Exception:
    # manual fallback
    text_emb_module = model.model.text_backbone.get_input_embeddings()
    new_emb_module = text_emb_module.resize_token_embeddings(len(tokenizer))
    model.model.text_backbone.set_input_embeddings(new_emb_module)

# 3) initialize the new prompt‐token embeddings from the [CLS] embedding
with torch.no_grad():
    cls_id = tokenizer.cls_token_id
    cls_emb = model.model.text_backbone.embeddings.word_embeddings.weight[cls_id]
    # the last NUM_PROMPT_TOK rows correspond to our newly added tokens
    model.model.text_backbone.embeddings.word_embeddings.weight[-NUM_PROMPT_TOK:] = cls_emb

# 4) freeze everything except the word‐embeddings matrix
for name, param in model.named_parameters():
    if "text_backbone.embeddings.word_embeddings.weight" not in name:
        param.requires_grad = False

# helper to prepend prompt‐token IDs to any batch of input_ids/masks
PROMPT_IDS = torch.arange(len(tokenizer) - NUM_PROMPT_TOK, len(tokenizer), device=device)
def prepend_prompts(input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
    B, L = input_ids.shape
    prefix_ids   = PROMPT_IDS.unsqueeze(0).expand(B, -1)                           # (B, P)
    prefix_mask  = torch.ones(B, NUM_PROMPT_TOK, device=device, dtype=attention_mask.dtype)
    new_input_ids   = torch.cat([prefix_ids, input_ids],   dim=1)  # (B, P+L)
    new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, P+L)
    return new_input_ids, new_attention_mask



# === Cell 8: Prefix‐Tuning Loop (fixed optimizer) ===

# 1) reference the full embedding matrix (leaf parameter)
embeddings = model.model.text_backbone.embeddings.word_embeddings

# 2) freeze everything except the full embedding matrix
for name, param in model.named_parameters():
    if "text_backbone.embeddings.word_embeddings.weight" not in name:
        param.requires_grad = False
    else:
        # this is our only leaf with requires_grad=True
        param.requires_grad = True

# 3) optimizer over the full embedding parameter
optimizer = torch.optim.AdamW(
    [embeddings.weight], 
    lr=LR_PROMPT
)

# 4) scheduler as before
num_steps = (NUM_EPOCHS * len(val_ds)) // BATCH_SIZE
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=num_steps
)

# 5) data loader
train_ds     = CocoInferenceDataset(TRAIN_ROOT, TRAIN_ANN, processor)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_inference
)

print("\n>>> Starting prefix‐tuning …")
model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        img_id, path, img = batch[0]

        # 6) tokenize + prepend prompt‐IDs
        inputs = processor(images=img, text=[baseline_prompt], return_tensors="pt").to(device)
        inputs.input_ids, inputs.attention_mask = prepend_prompts(
            inputs.input_ids, inputs.attention_mask
        )

        # 7) forward pass
        outputs = model(
            pixel_values=inputs.pixel_values,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

        # 8) loss = –mean(confidence)
        scores = outputs.logits.squeeze(-1)
        # debug check
        # print("score stats:", scores.min().item(), scores.max().item())
        loss   = -scores.mean()

        # 9) backward + step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg:.4f}")

# === Save the tuned model ===
# OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tuned_model")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Save the PyTorch weights
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "groundingdino_prefix_tuned.pth"))

# 2) (Optional) Save the tokenizer so prefix-IDs stay consistent
tokenizer.save_pretrained(OUTPUT_DIR)

# 3) Save any config or hyperparams for reproducibility
with open(os.path.join(OUTPUT_DIR, "tuning_config.json"), "w") as f:
    json.dump({
        "NUM_PROMPT_TOK": NUM_PROMPT_TOK,
        "LR_PROMPT": LR_PROMPT,
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
    }, f, indent=2)

print(f"📝 Tuned model and tokenizer saved to {OUTPUT_DIR}")



# === Cell 9: Reload & Evaluate Tuned Model ===

from transformers import BertTokenizer, AutoProcessor, AutoModelForZeroShotObjectDetection

# 1) Load tokenizer (with your added prompt tokens)
# SAVE_DIR = os.path.join(OUTPUT_DIR, "tuned_model")
SAVE_DIR = OUTPUT_DIR
tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)

# 2) Reinstantiate processor & model
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(device)

# 3) Resize & load your tuned embeddings
model.model.text_backbone.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "groundingdino_prefix_tuned.pth")))

model.eval()
print("✔️  Prompt-tuned model reloaded.")

# 4) Run the same eval as baseline
tuned_dets = run_zero_shot(
    model, processor, val_ds,
    prompt=baseline_prompt,
    box_thresh=0.4, text_thresh=0.25,
    batch_size=4,
    out_json=os.path.join(OUTPUT_DIR, "zs_tuned.json")
)

print("\n>>> Quantitative Eval of Tuned Model:")
_ = coco_eval_from_list(COCO(VAL_ANN), tuned_dets)



# === Cell 10: Visualization Comparison (fixed) ===

import random
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore

def compare_visuals(img_id, baseline_preds, tuned_preds, dataset, processor,
                    box_thresh=0.4, text_thresh=0.25):
    # Load image
    info     = dataset.coco.loadImgs(img_id)[0]
    img_path = os.path.join(dataset.img_root, info["file_name"])
    image    = Image.open(img_path).convert("RGB")
    
    # Post‐process baseline
    b_outs = baseline_preds[img_id]["outputs"]
    b_ids  = baseline_preds[img_id]["input_ids"]
    b_res  = processor.post_process_grounded_object_detection(
        b_outs, b_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.size[::-1]]
    )[0]
    
    # Post‐process tuned
    t_outs = tuned_preds[img_id]["outputs"]
    t_ids  = tuned_preds[img_id]["input_ids"]
    t_res  = processor.post_process_grounded_object_detection(
        t_outs, t_ids,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        target_sizes=[image.size[::-1]]
    )[0]
    
    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, res, title in zip(
        (ax1, ax2),
        (b_res, t_res),
        ("Zero‐Shot Baseline", "Prefix‐Tuned")
    ):
        ax.imshow(image)
        ax.set_title(title)
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            x0, y0, x1, y1 = box.cpu().tolist()
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                edgecolor="red", fill=False, linewidth=2
            ))
            ax.text(x0, y0, f"{label}:{score:.2f}",
                    bbox=dict(facecolor="yellow", alpha=0.5))
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def collect_for_visual(model, processor, ds, prompt, img_ids):
    """
    Runs inference on exactly the list img_ids and returns a dict mapping each
    img_id to its HF outputs and input_ids.
    """
    model.eval()
    out = {}
    with torch.no_grad():
        for img_id, path, img in ds:
            if img_id not in img_ids:
                continue
            inputs = processor(images=img, text=[prompt], return_tensors="pt").to(device)
            outputs = model(**inputs)
            out[img_id] = {
                "outputs": outputs,
                "input_ids": inputs.input_ids.cpu()
            }
            # free GPU memory immediately
            del outputs, inputs
            torch.cuda.empty_cache()
    return out


# 1) sample a fixed small subset
subset = random.sample(val_ds.ids, k=3)

# 2) collect baseline & tuned using the same subset
baseline_map = collect_for_visual(model, processor, val_ds, baseline_prompt, subset)
tuned_map    = collect_for_visual(model, processor, val_ds, baseline_prompt, subset)

# 3) visualize each
for img_id in subset:
    compare_visuals(img_id, baseline_map, tuned_map, val_ds, processor)