# SatQuery — Judge-facing feature sheet

| Capability | What the prototype actually does | Why it matters |
|---|---|---|
| Natural-language EO query | Maps questions to flood/water/vegetation/deforestation/burn/urban/infrastructure/agriculture intents | Removes GIS-first workflow friction |
| Multispectral intelligence | Reads six-band HLS-style GeoTIFFs and computes NDVI/NDWI/NBR | Uses physically meaningful spectral evidence |
| Before/After | Co-registered NDVI/NDWI/NBR transitions + change regions | Stronger than raw visual differencing |
| Prithvi | Optional real `prithvi_eo_v2_300` pretrained backbone forward pass | Foundation-model-ready architecture without fake task claims |
| Geospatial integrity | Real CRS/bounds for GeoTIFF; approximate label for JPG/PNG | Prevents misleading map coordinates |
| Evidence-first reports | Separates measured signals from interpretation and limitations | Better trust and auditability |
| GIS exports | GeoJSON / JSON / CSV | Easy handoff to QGIS/ArcGIS workflows |
| Secure access | Hashed passwords, JWT, protected APIs, rate limits, upload validation | Safer prototype deployment |
| SAR | Explicitly future/unimplemented | Avoids misleading radar claims |

### One-line uniqueness
**Ask a satellite question in plain language, get measurable spectral/change evidence, see it geospatially, and export the result — all from one dashboard.**
