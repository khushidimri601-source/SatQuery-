# SatQuery AI

**Natural-language Earth Observation intelligence dashboard** for scene screening, multispectral evidence, Before/After change analysis, timelines, reports and GIS-ready exports.

## What is actually implemented
- **RGB screening:** JPG/PNG → deterministic visual metrics + candidate regions.
- **Real multispectral GeoTIFF:** six-band HLS-style input → NDVI, NDWI, NBR, real CRS/bounds and spectral candidate regions.
- **Stronger Before/After:** common-grid multispectral comparison detects measurable NDVI/NDWI/NBR transitions; RGB fallback remains available for ordinary images.
- **Prithvi-EO-2.0:** optional real pretrained `prithvi_eo_v2_300` backbone forward pass on compatible six-band GeoTIFFs via TerraTorch.
- **Timeline:** multi-scene change trends and evidence.
- **Reports + exports:** structured report, GeoJSON/JSON/CSV.
- **Authentication:** bcrypt password hashing, JWT sessions, registration, protected APIs, rate limiting and upload validation.
- **SAR:** deliberately marked future/unimplemented; no simulated backscatter claims.

## Why this is stronger
SatQuery separates **evidence from interpretation**. RGB outputs are labelled as screening. Multispectral outputs use physical spectral indices. Prithvi is used only when compatible six-band GeoTIFF data and the optional model stack are available.

## Install — Windows
```powershell
npm install
py -m pip install -r requirements.txt
copy .env.example .env
# edit .env and set a long random JWT_SECRET
npm start
```
Open `http://localhost:3000`. Create an analyst account on the sign-in screen.

### Optional Prithvi
```powershell
py -m pip install -r requirements-prithvi.txt
```
Set `SATQUERY_PRITHVI_ENABLED=true` in `.env`, restart the server, and upload a compatible six-band HLS-style GeoTIFF. The first model run can download large pretrained weights and may require a GPU for practical speed.

### Real sample imagery
```powershell
py download_sample_data.py
```
The downloader references public NASA Earth Observatory scenes for flood, deforestation, urban growth and agriculture.

## Data contract for Prithvi
Use georeferenced HLS-style GeoTIFFs with six bands in this order:
1. Blue
2. Green
3. Red
4. Narrow NIR
5. SWIR 1
6. SWIR 2

## Limitations
- RGB screening is not a validated semantic model.
- Prithvi backbone embeddings alone are not task-specific labels.
- Flood/vegetation/burn interpretations are evidence signals, not guaranteed ground truth.
- Before/After scenes should be co-registered and cloud/nodata screened.
- Large GeoTIFFs are downsampled for interactive analysis; use production tiling for very large scenes.

## Judge-facing positioning
> **SatQuery turns a natural-language EO question into evidence-backed geospatial analysis — combining fast RGB screening, real multispectral indices, temporal change evidence, optional Prithvi foundation-model inference, and GIS-ready outputs in one workflow.**


## Scientific guardrails

- **Evidence score is not accuracy.** SatQuery reports measurable signal strength/region consistency; it does not claim a universal detection probability.
- **Flood:** a single scene reports a water/inundation candidate. A flood claim requires temporal evidence of newly appearing water and should be checked against cloud/nodata quality.
- **Vegetation loss:** a single scene reports low-vegetation candidates. Forest-loss/deforestation interpretation is temporal and contextual.
- **Before/After:** compatible GeoTIFF scenes are automatically reprojected/resampled onto the After grid before multispectral comparison.
- **Prithvi:** the integrated Prithvi-EO-2.0 path is a real pretrained foundation-model backbone forward pass. It is not presented as a task-specific flood/deforestation detector without a downstream checkpoint. Prithvi's pretraining uses HLS-style multispectral inputs and TerraTorch supports downstream fine-tuning.
- **SAR:** not implemented in this build. It is intentionally shown as future capability rather than simulated or claimed functionality.
- **Band contract:** true multispectral mode requires six bands in Prithvi/HLS order: Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2. Sentinel-2 red-edge B05/B06/B07 are not silently treated as those bands.

## Security

Create `.env` and set a random `JWT_SECRET` of at least 32 characters. The server refuses to start with a missing or weak secret. Authentication uses bcrypt password hashing and short-lived JWT sessions; analysis endpoints require authentication. Upload size/type limits and API rate limiting are enabled.
