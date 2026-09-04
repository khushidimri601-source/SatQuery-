# SatQuery AI — Satellite AI Model Selection & Fine-Tuning Guide

This document provides a technical roadmap for selecting, fine-tuning, and deploying Deep Learning Vision-Language Models (VLMs) and Object Detectors for Earth Observation (EO) and geospatial satellite analysis.

---

## 1. AI Model Selection Architecture Matrix

Depending on your target geospatial application, select the optimal architecture below:

| Application Goal | Model Architecture | Core Strengths | Target Benchmarks / Datasets |
| :--- | :--- | :--- | :--- |
| **Vision-Language Query & Grounding** | **GeoChat / LLaVA-Geo** | Multi-modal VLM pre-trained on high-res satellite imagery. Accepts natural language queries and outputs bounding polygons. | **RSVQA-HR**, **GeoGrounded**, **SkyScript** |
| **Foundation Segmentation** | **SAM-Geo (Segment Anything for Geospatial)** | Zero-shot / few-shot geospatial feature extraction (floods, water, forests, urban structures, agricultural fields). | **SpaceNet 1-7**, **Inria Aerial Footprint** |
| **Fast Object Detection** | **YOLOv8-OBB / YOLOv11-OBB** | Oriented Bounding Box detection for arbitrary-rotated satellite targets (ships, planes, solar farms, vehicles). | **DOTA v2.0**, **NWPU VISC-10** |
| **Multispectral / SAR Encoder** | **Prithvi-EO / RemoteCLIP** | Vision Transformer foundation backbones by NASA/IBM for multi-band Sentinel-2 & SAR imagery. | **Sentinel-2 L2A**, **Landsat 8/9** |

---

## 2. Dataset Preparation & Preprocessing

### A. Vision-Language Dataset Format (`satquery_vlm_dataset.json`)
```json
[
  {
    "id": "sample_001",
    "image_path": "tiles/sentinel2_assam_01.tif",
    "query": "Find all flooded regions in this scene",
    "target_class": "flood",
    "bboxes": [
      [94.175, 26.955, 94.185, 26.965]
    ],
    "ground_truth_geojson": {
      "type": "Polygon",
      "coordinates": [[[94.175, 26.955], [94.185, 26.955], [94.185, 26.965], [94.175, 26.965], [94.175, 26.955]]]
    }
  }
]
```

### B. Standard Satellite Datasets to Download
1. **RSVQA (Remote Sensing Visual Question Answering)**: 21,000+ low & high resolution images with question-answer-bounding box triplets.
2. **SpaceNet Building & Road Extraction**: Multi-city Sentinel-2 and DigitalGlobe high-resolution GeoTIFFs.
3. **DOTA v2.0**: 11,268 satellite scenes with 1.7 million oriented bounding box instances.

---

## 3. Training & Fine-Tuning Execution

Run the custom PyTorch fine-tuning script provided in `train_satquery_vlm.py`:

```bash
python train_satquery_vlm.py \
  --data_json ./data/satquery_train.json \
  --image_dir ./data/satellite_tiles \
  --epochs 15 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir ./checkpoints/satquery_vlm_v1
```

---

## 4. Deploying Checkpoints to the Backend

Once training completes, place the saved PyTorch model (`satquery_vlm_best.pth` or ONNX export) into the backend directory. `ai_model_bridge.py` can load `torch.jit` or `onnxruntime` models for live inference.
