# SatQuery model roadmap

## Current production prototype
SatQuery currently has three honest analysis layers:

1. **RGB screening** for JPG/PNG using deterministic image signals.
2. **Multispectral evidence** for six-band HLS-style GeoTIFFs using NDVI, NDWI and NBR.
3. **Optional Prithvi-EO-2.0 backbone inference** for compatible six-band GeoTIFFs through TerraTorch.

The Prithvi backbone is a foundation encoder. It does **not** magically turn into a flood/deforestation detector. A task-specific downstream checkpoint is required for validated semantic segmentation/classification.

## Recommended downstream tasks

| Task | Recommended path | Output |
|---|---|---|
| Flood mapping | Prithvi + a flood segmentation checkpoint | Pixel-level flood mask |
| Vegetation/crop monitoring | Prithvi + crop/land-cover checkpoint | Crop/land-cover map |
| Burn scars | Prithvi + HLS burn-scar checkpoint | Burn-scar mask |
| Change detection | Co-registered multispectral indices + optional learned model | Change mask + evidence |
| Natural-language grounding | Separate EO VLM such as GeoChat-style research model | Query-to-region explanation |

NASA's Prithvi-EO-2.0 repository provides TerraTorch configurations for flood detection, burn scars and multi-temporal crop classification. SatQuery should use those as the starting point for future task-specific checkpoints rather than claiming the base backbone itself performs those tasks.

## Experimental VLM training script
`train_satquery_vlm.py` is retained only as an **Experimental VLM Training Prototype**. It is not the production inference path and must not be described as a trained SatQuery foundation model.

## Evaluation plan before claiming accuracy
For any task-specific model:

- keep a held-out geographic test set;
- report IoU/F1/precision/recall, not a generic “AI accuracy”;
- test across seasons, cloud conditions and locations;
- compare against a simple baseline;
- document class definitions and no-data/cloud handling.

## Data contract
Prithvi-EO-2.0 expects HLS-style six-band inputs in this order:
`BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`.
For temporal/location-aware variants, preserve acquisition date and scene geolocation metadata.

## Official references
- NASA/IMPACT Prithvi-EO-2.0: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- Prithvi-EO-2.0 models: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
- TerraTorch documentation: https://torchgeo.org/terratorch/1.2.10/guide/quick_start/
