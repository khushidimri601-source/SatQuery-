"""
SatQuery AI — Experimental VLM Training Prototype
-------------------------------------------------------------------------
This module is an experimental training prototype for Earth Observation (EO)
vision-language spatial grounding. It is NOT a production fine-tuning pipeline.

It accepts remote sensing imagery tiles paired with natural language prompt queries
and trains a spatial bounding predictor.
"""

import os
import json
import argparse
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------
# 1. Custom Remote Sensing VLM PyTorch Dataset
# ----------------------------------------------------
class SatQueryVLDataset(Dataset):
    def __init__(self, json_path: str, image_dir: str):
        self.image_dir = image_dir
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Simulate loading satellite image tile (C, H, W)
        # In full pipeline, use rasterio / GDAL for GeoTIFF multispectral loading
        dummy_tile = torch.randn(3, 256, 256) 
        
        # Convert query prompt to synthetic text encoding vector
        query_text = sample.get('query', '')
        query_vec = torch.zeros(128)
        for i, char in enumerate(query_text[:128]):
            query_vec[i] = ord(char) / 255.0
            
        # Target bounding box normalized coordinates [ymin, xmin, ymax, xmax]
        bboxes = sample.get('bboxes', [[0.2, 0.2, 0.5, 0.5]])
        target_box = torch.tensor(bboxes[0], dtype=torch.float32)
        
        return {
            'image': dummy_tile,
            'query_vec': query_vec,
            'target_box': target_box,
            'query_text': query_text
        }

# ----------------------------------------------------
# 2. Vision-Language Spatial Grounding Network Architecture
# ----------------------------------------------------
class SatQueryVLM(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super(SatQueryVLM, self).__init__()
        
        # Vision Encoder BackBone (Simulated ResNet/ViT feature extractor)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Language Encoder Projection
        self.lang_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Vision-Language Multimodal Fusion & Box Regression Head
        self.fusion_head = nn.Sequential(
            nn.Linear(128 * 8 * 8 + 128, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid() # Normalize coordinates [0, 1]
        )
        
    def forward(self, images, query_vecs):
        batch_size = images.size(0)
        v_feats = self.vision_encoder(images) # [B, 128, 8, 8]
        v_feats_flat = v_feats.view(batch_size, -1) # [B, 128*8*8]
        
        l_feats = self.lang_encoder(query_vecs) # [B, 128]
        
        # Concatenate Vision + Language embeddings
        fused = torch.cat([v_feats_flat, l_feats], dim=1)
        pred_boxes = self.fusion_head(fused)
        return pred_boxes

# ----------------------------------------------------
# 3. Smooth L1 + IoU Loss Function
# ----------------------------------------------------
class SpatialGroundingLoss(nn.Module):
    def __init__(self):
        super(SpatialGroundingLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()
        
    def forward(self, pred_boxes, target_boxes):
        loss_l1 = self.l1_loss(pred_boxes, target_boxes)
        return loss_l1

# ----------------------------------------------------
# 4. Training Loop Execution
# ----------------------------------------------------
def train_satquery_vlm(args):
    print("==========================================================")
    print("🛰️ Starting SatQuery AI Experimental VLM Training Prototype")
    print(f"📁 Dataset JSON: {args.data_json}")
    print(f"⚙️ Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("==========================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Computing Hardware: {device}")
    
    # Initialize Dataset & DataLoader
    if os.path.exists(args.data_json):
        dataset = SatQueryVLDataset(args.data_json, args.image_dir)
    else:
        print(f"⚠️ Dataset JSON file '{args.data_json}' not found. Generating dummy dataset for training run.")
        dummy_data = [
            {"id": f"sample_{i}", "query": "Detect flood extent", "bboxes": [[0.1, 0.2, 0.6, 0.7]]}
            for i in range(32)
        ]
        os.makedirs(os.path.dirname(args.data_json) or '.', exist_ok=True)
        with open(args.data_json, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        dataset = SatQueryVLDataset(args.data_json, args.image_dir)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SatQueryVLM().to(device)
    criterion = SpatialGroundingLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for batch in dataloader:
            images = batch['image'].to(device)
            query_vecs = batch['query_vec'].to(device)
            target_boxes = batch['target_box'].to(device)
            
            optimizer.zero_grad()
            preds = model(images, query_vecs)
            loss = criterion(preds, target_boxes)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataset)
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.5f} - Time: {elapsed:.2f}s")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(args.output_dir, 'satquery_vlm_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  🏆 Saved new best checkpoint -> {ckpt_path}")
            
    print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery Experimental VLM Training Prototype")
    parser.add_argument("--data_json", type=str, default="./data/satquery_train.json")
    parser.add_argument("--image_dir", type=str, default="./data/tiles")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    
    args = parser.parse_args()
    train_satquery_vlm(args)
