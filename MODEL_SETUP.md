# Model & data setup

## 1. RGB path
JPG/PNG uploads use deterministic computer-vision screening. This is intentionally not described as a trained satellite VLM.

## 2. Real multispectral path
For six-band HLS-style GeoTIFFs, SatQuery reads:
`BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`
and computes georeferenced NDVI, NDWI and NBR evidence. Before/After scenes must share a common CRS and pixel grid.

## 3. Real Prithvi-EO-2.0 backbone
The optional integration uses TerraTorch's `prithvi_eo_v2_300` pretrained backbone. This is a real forward pass, not a mock/random model. Prithvi-EO-2.0 was pretrained by NASA/IBM/Jülich on HLS time-series data with six bands.

Install:
```powershell
py -m pip install -r requirements-prithvi.txt
```
Then in `.env`:
```text
SATQUERY_PRITHVI_ENABLED=true
SATQUERY_PRITHVI_MODEL=prithvi_eo_v2_300
```

Important: a pretrained backbone is not automatically a flood/deforestation detector. Task-specific semantic predictions require a compatible downstream segmentation/classification checkpoint. SatQuery therefore combines the backbone with explicit spectral evidence instead of inventing task accuracy.

## 4. SAR
SAR is **not implemented** in this build. The UI marks it as future capability. No SAR backscatter is simulated or claimed.
