# SatQuery change detection

## RGB Before/After
For JPG/PNG scenes, SatQuery performs:
- size alignment;
- brightness/contrast normalization;
- per-pixel RGB difference;
- noise cleanup and connected components;
- changed-area percentage and visual heatmap.

This is evidence of **visual change**, not a semantic guarantee.

## Multispectral Before/After
For compatible six-band HLS-style GeoTIFFs, SatQuery uses a stronger path:
- verifies CRS and pixel-grid alignment;
- computes NDVI, NDWI and NBR for each date;
- measures spectral transitions;
- reports water/inundation, vegetation-loss and burn-scar signals;
- returns real georeferenced change regions.

Interpretation is explicit:
- **Flood signal:** positive NDWI transition.
- **Vegetation-loss signal:** negative NDVI transition.
- **Burn signal:** negative NBR transition.

These are measurable spectral signals, not ground-truth labels. Cloud, shadow, seasonal effects and imperfect registration can still create false changes.

## Best practice
Use co-registered scenes from the same sensor/product where possible, preserve acquisition dates, and apply cloud/nodata masks before operational use.
