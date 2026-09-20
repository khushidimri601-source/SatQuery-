"""Real multispectral/HLS analysis for SatQuery.

Supports six-band HLS-style GeoTIFFs in the Prithvi order:
Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2.
Computes physical spectral indices (NDVI, NDWI, NBR), georeferenced
candidate regions, and optional Prithvi-EO-2.0 backbone embeddings.
"""
from __future__ import annotations
import json, os, base64, io
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np

BAND_NAMES = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
HLS_MEAN = np.array([1087, 1342, 1433, 2734, 1958, 1363], dtype=np.float32)
HLS_STD = np.array([2248, 2179, 2178, 1850, 1242, 1049], dtype=np.float32)


def is_geotiff(path):
    return str(path).lower().endswith((".tif", ".tiff"))


def _open(path):
    import rasterio
    return rasterio.open(path)


def _band_map(src):
    desc = [str(x or "").upper() for x in (src.descriptions or [])]
    aliases = {
        "BLUE": ["BLUE", "B02", "BAND2", "B2"],
        "GREEN": ["GREEN", "B03", "BAND3", "B3"],
        "RED": ["RED", "B04", "BAND4", "B4"],
        "NIR_NARROW": ["NIR_NARROW", "NIR_NARROW_BAND", "NIR8A", "B8A", "BAND8A", "B8A_BAND"],
        "SWIR_1": ["SWIR_1", "SWIR1", "B11", "BAND11"],
        "SWIR_2": ["SWIR_2", "SWIR2", "B12", "BAND12"],
    }
    out = {}
    for i, d in enumerate(desc, 1):
        clean = d.replace(" ", "_").replace("-", "_")
        for name, vals in aliases.items():
            if any(v == clean or v in clean for v in vals):
                out.setdefault(name, i)
    # Unlabelled six-band rasters are accepted only as the documented HLS/Prithvi contract.
    # We deliberately do NOT guess Sentinel-2 B05/B06/B07 mappings because those are red-edge
    # bands and would be scientifically wrong for Prithvi's NIR_NARROW/SWIR inputs.
    if not any(desc) and src.count == 6:
        out = {name: i + 1 for i, name in enumerate(BAND_NAMES)}
        contract = "assumed_hls_six_band_order"
    else:
        contract = "named_bands"
    return out, contract


def read_hls(path, max_pixels=2_000_000):
    with _open(path) as src:
        mapping, band_contract = _band_map(src)
        missing = [b for b in BAND_NAMES if b not in mapping]
        if missing:
            raise ValueError(f"GeoTIFF is missing required HLS/Prithvi bands: {', '.join(missing)}")
        scale = src.scales[0] if src.scales else 1.0
        offset = src.offsets[0] if src.offsets else 0.0
        arr = np.stack([src.read(mapping[b]).astype(np.float32) * scale + offset for b in BAND_NAMES])
        if arr.shape[1] * arr.shape[2] > max_pixels:
            factor = (max_pixels / (arr.shape[1] * arr.shape[2])) ** 0.5
            from rasterio.enums import Resampling
            h = max(32, int(arr.shape[1] * factor)); w = max(32, int(arr.shape[2] * factor))
            arr = src.read([mapping[b] for b in BAND_NAMES], out_shape=(6, h, w), resampling=Resampling.bilinear).astype(np.float32)
            arr *= scale; arr += offset
        nodata = src.nodata
        valid = np.all(np.isfinite(arr), axis=0)
        if nodata is not None:
            valid &= np.all(arr != nodata, axis=0)
        # HLS reflectance is commonly stored as 0..10000 integer-scaled values.
        if np.nanpercentile(np.abs(arr[:, valid]), 99) > 20:
            refl = arr / 10000.0
        else:
            refl = arr
        return refl, valid, src.profile.copy(), src.transform, src.crs, mapping, band_contract


def ratio(a, b):
    den = a + b
    return np.divide(a - b, den, out=np.zeros_like(a), where=np.abs(den) > 1e-6)


def indices(refl, valid):
    blue, green, red, nir, swir1, swir2 = refl
    ndvi = ratio(nir, red)
    ndwi = ratio(green, nir)
    nbr = ratio(nir, swir2)
    for x in (ndvi, ndwi, nbr):
        x[~valid] = np.nan
    return {"ndvi": ndvi, "ndwi": ndwi, "nbr": nbr}


def summary(refl, valid):
    idx = indices(refl, valid)
    def pct(mask): return round(float(np.nanmean(mask[valid]) * 100), 2) if valid.any() else 0.0
    ndvi, ndwi, nbr = idx["ndvi"], idx["ndwi"], idx["nbr"]
    return {
        "format": "GeoTIFF/HLS multispectral",
        "bands": BAND_NAMES,
        "valid_pixel_percent": round(float(valid.mean() * 100), 2),
        "ndvi_mean": round(float(np.nanmean(ndvi)), 4),
        "ndvi_vegetation_percent": pct(ndvi > 0.35),
        "ndwi_mean": round(float(np.nanmean(ndwi)), 4),
        "ndwi_water_percent": pct(ndwi > 0.25),
        "nbr_mean": round(float(np.nanmean(nbr)), 4),
        "burn_scar_signal_percent": pct(nbr < 0.15),
    }


def _components(mask, min_area):
    from scipy import ndimage
    labels, count = ndimage.label(mask)
    out=[]
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        if sl is None: continue
        area=int((labels[sl]==i).sum())
        if area < min_area: continue
        out.append((area, sl[1].start, sl[0].start, sl[1].stop-1, sl[0].stop-1))
    return sorted(out, reverse=True)


def _geo_features(mask, transform, crs, label, signal_name, min_area=80):
    from rasterio.warp import transform_bounds
    from affine import Affine
    import rasterio.features
    h,w=mask.shape
    comps=_components(mask,min_area)[:15]
    out=[]
    for i,(area,x1,y1,x2,y2) in enumerate(comps,1):
        left,top = transform * (x1,y1)
        right,bottom = transform * (x2+1,y2+1)
        if crs:
            west,south,east,north=transform_bounds(crs,"EPSG:4326",left,bottom,right,top)
        else:
            west,south,east,north=left,bottom,right,top
        ring=[[west,south],[east,south],[east,north],[west,north],[west,south]]
        conf=int(np.clip(60 + min(35, area/(h*w)*7000), 60, 95))
        out.append({"type":"Feature","properties":{"id":f"ms_{signal_name}_{i}","class":label,"label":label,"confidence":conf,"source":"spectral-index","signal":signal_name,"coordinate_type":"georeferenced"},"geometry":{"type":"Polygon","coordinates":[ring]}})
    return out


def analyze(path, query_class="generic"):
    refl, valid, profile, transform, crs, mapping, band_contract = read_hls(path)
    s=summary(refl,valid); idx=indices(refl,valid)
    q=query_class
    if q in ("flood","water"):
        mask=(idx["ndwi"] > 0.25) & valid
        label="Spectral water / inundation candidate"
        signal="NDWI"
    elif q in ("vegetation","agriculture"):
        mask=(idx["ndvi"] > 0.35) & valid
        label="Vegetation / crop candidate"
        signal="NDVI"
    elif q=="deforest":
        mask=(idx["ndvi"] < 0.2) & valid
        label="Low-vegetation candidate; confirm with before/after"
        signal="NDVI-low"
    elif q=="burn":
        mask=(idx["nbr"] < 0.15) & valid
        label="Burn-scar spectral candidate"
        signal="NBR"
    else:
        mask=((idx["ndvi"] > 0.35) | (idx["ndwi"] > 0.25)) & valid
        label="Multispectral candidate region"
        signal="NDVI/NDWI"
    feats=_geo_features(mask,transform,crs,label,signal)
    return {"summary":s,"features":feats,"crs":str(crs) if crs else None,"transform":list(transform) if transform else None,"band_map":mapping,"band_contract":band_contract,"shape":[int(refl.shape[1]),int(refl.shape[2])],"indices":{k:{"mean":round(float(np.nanmean(v)),4)} for k,v in idx.items()}}


def _preview(refl):
    # Natural-colour preview from Red/Green/Blue bands.
    rgb=np.clip(np.transpose(refl[[2,1,0]],(1,2,0)),0,1)
    # Robust contrast stretch for display only.
    lo=np.nanpercentile(rgb,2,axis=(0,1)); hi=np.nanpercentile(rgb,98,axis=(0,1)); rgb=(rgb-lo)/(hi-lo+1e-6)
    im=Image.fromarray(np.clip(rgb*255,0,255).astype(np.uint8)).resize((min(900,rgb.shape[1]),min(900,rgb.shape[0])))
    buf=io.BytesIO(); im.save(buf,format="JPEG",quality=82); return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")

def _change_preview(refl, change):
    rgb=np.clip(np.transpose(refl[[2,1,0]],(1,2,0)),0,1)
    lo=np.nanpercentile(rgb,2,axis=(0,1)); hi=np.nanpercentile(rgb,98,axis=(0,1)); rgb=np.clip((rgb-lo)/(hi-lo+1e-6),0,1)
    base=(rgb*255).astype(np.uint8); overlay=base.copy(); overlay[change]=[235,55,70]
    out=(0.68*base+0.32*overlay).astype(np.uint8)
    im=Image.fromarray(out); buf=io.BytesIO(); im.save(buf,format="JPEG",quality=82); return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")

def _scene_metadata(path):
    try:
        with _open(path) as src:
            tags = src.tags()
            date = tags.get("ACQUISITION_DATE") or tags.get("DATE_ACQUIRED") or tags.get("SENSING_TIME") or tags.get("datetime") or tags.get("TIFFTAG_DATETIME")
            return {"acquisition_date": str(date) if date else None, "crs": str(src.crs) if src.crs else None, "resolution": [abs(float(src.transform.a)), abs(float(src.transform.e))]}
    except Exception:
        return {"acquisition_date": None, "crs": None, "resolution": None}


def _align_to_after(before, valid_before, tb, crsb, after, valid_after, ta, crsa):
    """Reproject/resample Before onto the exact After pixel grid when needed."""
    if crsb == crsa and before.shape == after.shape and tb == ta:
        return before, valid_before, False
    from rasterio.warp import reproject, Resampling
    aligned = np.full_like(after, np.nan, dtype=np.float32)
    aligned_valid = np.zeros(after.shape[1:], dtype=bool)
    for i in range(after.shape[0]):
        reproject(
            source=before[i], destination=aligned[i],
            src_transform=tb, src_crs=crsb,
            dst_transform=ta, dst_crs=crsa,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    reproject(
        source=valid_before.astype(np.uint8), destination=aligned_valid.astype(np.uint8),
        src_transform=tb, src_crs=crsb,
        dst_transform=ta, dst_crs=crsa,
        src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
    )
    # The uint8 destination above is a temporary object; recompute a robust valid mask.
    aligned_valid = np.all(np.isfinite(aligned), axis=0) & valid_after
    return aligned, aligned_valid, True


def compare(before_path, after_path):
    rb,vb,pb,tb,crsb,mb,cb=read_hls(before_path)
    ra,va,pa,ta,crsa,ma,ca=read_hls(after_path)
    if not crsb or not crsa:
        raise ValueError("Both Before and After GeoTIFFs must be georeferenced for multispectral change analysis.")
    rb,vb,was_aligned=_align_to_after(rb,vb,tb,crsb,ra,va,ta,crsa)
    ib=indices(rb,vb); ia=indices(ra,va); valid=vb&va
    ndvi_delta=ia["ndvi"]-ib["ndvi"]; ndwi_delta=ia["ndwi"]-ib["ndwi"]; nbr_delta=ia["nbr"]-ib["nbr"]
    # Temporal evidence: new inundation / vegetation loss / burn-scar transitions.
    flood=(ndwi_delta>0.20)&valid
    veg_loss=(ndvi_delta<-0.20)&valid
    burn=(nbr_delta<-0.20)&valid
    change=(np.abs(ndvi_delta)>0.18)|(np.abs(ndwi_delta)>0.18)|(np.abs(nbr_delta)>0.18)
    change &= valid
    def pct(m): return round(float(m[valid].mean()*100),2) if valid.any() else 0.0
    comps=_components(change, max(80, int(change.size*0.00012)))[:20]
    regions=[]
    from rasterio.warp import transform_bounds
    for i,(area,x1,y1,x2,y2) in enumerate(comps,1):
        left,top=ta*(x1,y1); right,bottom=ta*(x2+1,y2+1)
        west,south,east,north=transform_bounds(crsa,"EPSG:4326",left,bottom,right,top)
        local_ndvi=np.abs(ndvi_delta[y1:y2+1,x1:x2+1])
        score=float(np.nanmean(local_ndvi)) if np.isfinite(local_ndvi).any() else 0.0
        evidence_score=int(np.clip(55 + score*140 + min(25, area/change.size*5000), 55, 98))
        regions.append({"id":f"change_{i}","x":x1/change.shape[1],"y":y1/change.shape[0],"width":(x2-x1+1)/change.shape[1],"height":(y2-y1+1)/change.shape[0],"area_percent":round(area/change.size*100,3),"evidence_score":evidence_score,"west":west,"south":south,"east":east,"north":north})
    before_meta=_scene_metadata(before_path); after_meta=_scene_metadata(after_path)
    out={
      "method":"co-registered multispectral index change detection",
      "coordinate_type":"georeferenced",
      "crs":str(crsa),
      "co_registration":{"performed":was_aligned,"note":"Before was resampled/reprojected onto the After grid before spectral comparison." if was_aligned else "Input grids already matched."},
      "change_percent":pct(change),
      "flood_signal_percent":pct(flood),
      "vegetation_loss_percent":pct(veg_loss),
      "burn_signal_percent":pct(burn),
      "regions":len(regions),
      "regions_data":regions,
      "before_preview":_preview(rb), "after_preview":_preview(ra), "change_preview":_change_preview(ra,change),
      "before":{"ndvi_mean":round(float(np.nanmean(ib["ndvi"])),4),"ndwi_mean":round(float(np.nanmean(ib["ndwi"])),4),"nbr_mean":round(float(np.nanmean(ib["nbr"])),4),**before_meta},
      "after":{"ndvi_mean":round(float(np.nanmean(ia["ndvi"])),4),"ndwi_mean":round(float(np.nanmean(ia["ndwi"])),4),"nbr_mean":round(float(np.nanmean(ia["nbr"])),4),**after_meta},
      "deltas":{"ndvi":round(float(np.nanmean(ndvi_delta[valid])),4),"ndwi":round(float(np.nanmean(ndwi_delta[valid])),4),"nbr":round(float(np.nanmean(nbr_delta[valid])),4)},
      "evidence":["New inundation signal = positive NDWI transition; permanent water is not automatically called a flood.","Vegetation-loss signal = negative NDVI transition; deforestation requires temporal/contextual confirmation.","Burn-scar evidence = negative NBR transition.","Cloud/nodata screening and temporal co-registration are required for reliable interpretation."],
      "limitations":["Spectral thresholds are evidence rules, not validated universal accuracy metrics.","Very different sensors/resolutions may reduce comparability even after resampling."]
    }
    return out


def prithvi_embedding(path, model_name="prithvi_eo_v2_300"):
    """Run a real Prithvi-EO-2.0 backbone forward pass when TerraTorch is installed."""
    try:
        import torch
        from terratorch import BACKBONE_REGISTRY
    except Exception as exc:
        return {"enabled":False,"error":"Install requirements-prithvi.txt to enable real Prithvi inference.","detail":str(exc)}
    refl,valid,_,_,_,_,_=read_hls(path,max_pixels=224*224)
    # Prithvi HLS normalization uses the six-band order from its HLS pretraining.
    x=np.nan_to_num(refl, nan=0.0001) * 10000.0
    x=(x-HLS_MEAN[:,None,None])/HLS_STD[:,None,None]
    x=torch.from_numpy(x).unsqueeze(0).float()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=BACKBONE_REGISTRY.build(model_name,bands=BAND_NAMES,num_frames=1,pretrained=True)
    model.to(device).eval(); x=x.to(device)
    with torch.no_grad(): out=model(x)
    if isinstance(out,(list,tuple)): tensors=[o for o in out if hasattr(o,"shape")]
    else: tensors=[out]
    stats=[]
    for t in tensors:
        z=t.detach().float()
        stats.append({"shape":list(z.shape),"mean":round(float(z.mean().cpu()),6),"std":round(float(z.std().cpu()),6)})
    return {"enabled":True,"model":model_name,"device":str(device),"output_layers":stats,"note":"Real pretrained Prithvi backbone forward pass on six-band HLS-style GeoTIFF."}
