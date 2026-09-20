"""Real before/after satellite image change detection using image differencing.

This is intentionally lightweight: it aligns both images to the same size, normalizes
brightness, computes per-pixel RGB difference, thresholds changed regions, removes
small noise, finds connected components, and returns a heatmap plus bounding boxes.
It does NOT pretend to be a semantic VLM detector.
"""
import argparse
import base64
import io
import json
import os
from collections import deque

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Reuse SatQuery's deterministic scene metrics for a textual explanation of change.
try:
    from ai_model_bridge import metrics as scene_metrics
except Exception:
    scene_metrics = None
try:
    import multispectral_analysis as ms
except Exception:
    ms = None


def load_image(path, max_side=1400):
    img = Image.open(path).convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    return img


def normalize_pair(a, b):
    size = (min(a.width, b.width), min(a.height, b.height))
    a = a.resize(size, Image.Resampling.LANCZOS)
    b = b.resize(size, Image.Resampling.LANCZOS)
    aa = np.asarray(a).astype(np.float32)
    bb = np.asarray(b).astype(np.float32)
    # Match global brightness/contrast so lighting/exposure shifts are less likely to be flagged.
    for c in range(3):
        am, asd = aa[:, :, c].mean(), aa[:, :, c].std() + 1e-6
        bm, bsd = bb[:, :, c].mean(), bb[:, :, c].std() + 1e-6
        bb[:, :, c] = np.clip((bb[:, :, c] - bm) * (asd / bsd) + am, 0, 255)
    return a, Image.fromarray(bb.astype(np.uint8))


def binary_cleanup(mask, iterations=1):
    m = mask.copy()
    h, w = m.shape
    for _ in range(iterations):
        padded = np.pad(m, 1)
        count = sum(padded[dy:dy+h, dx:dx+w] for dy in range(3) for dx in range(3))
        m = count >= 2
    return m


def components(mask, min_area):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = [(y, x)]
            seen[y, x] = True
            n = 0
            minx = maxx = x
            miny = maxy = y
            while q:
                cy, cx = q.pop()
                n += 1
                minx, maxx = min(minx, cx), max(maxx, cx)
                miny, maxy = min(miny, cy), max(maxy, cy)
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if n >= min_area:
                out.append((n, minx, miny, maxx, maxy))
    return sorted(out, reverse=True)


def data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def compare(before_path, after_path, sensitivity=35):
    # Preferred path for real HLS/Prithvi-compatible GeoTIFFs. It requires the same
    # geospatial grid so pixel-to-pixel change is physically meaningful.
    if ms is not None and ms.is_geotiff(before_path) and ms.is_geotiff(after_path):
        try:
            result = ms.compare(before_path, after_path)
            result["status"] = "success"
            result["sensitivity"] = sensitivity
            result["method"] = "co-registered six-band spectral change (NDVI/NDWI/NBR)"
            return result
        except Exception as exc:
            # Fall back smoothly to visual image comparison if GeoTIFF co-registration or multispectral read fails
            pass

    before = load_image(before_path)
    after = load_image(after_path)
    before, after = normalize_pair(before, after)
    a = np.asarray(before).astype(np.float32)
    b = np.asarray(after).astype(np.float32)

    # Mean absolute RGB difference, then light blur to suppress single-pixel noise.
    diff = np.mean(np.abs(a - b), axis=2)
    diff_img = Image.fromarray(np.clip(diff * 3, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    diff = np.asarray(diff_img).astype(np.float32) / 3.0

    # User sensitivity is 1..100: higher means more sensitive to smaller changes.
    threshold = float(np.clip(55 - sensitivity * 0.35, 8, 55))
    mask = diff >= threshold
    mask = binary_cleanup(mask, 2)
    min_area = max(40, int(mask.size * 0.00012))
    comps = components(mask, min_area)

    h, w = mask.shape
    changed_pixels = int(mask.sum())
    change_percent = round(changed_pixels / (w * h) * 100, 2)

    boxes = []
    for i, (area, x1, y1, x2, y2) in enumerate(comps[:20], 1):
        boxes.append({
            "id": f"change_{i}",
            "x": x1 / w,
            "y": y1 / h,
            "width": (x2 - x1 + 1) / w,
            "height": (y2 - y1 + 1) / h,
            "area_percent": round(area / (w * h) * 100, 3),
            "change_score": round(float(diff[y1:y2+1, x1:x2+1].mean()), 1)
        })

    # Heatmap overlay: changed pixels are highlighted while the after image remains visible.
    base = np.asarray(after).astype(np.float32)
    norm = np.clip((diff - threshold) / max(1, 100 - threshold), 0, 1)
    heat = np.zeros_like(base)
    heat[:, :, 0] = 255
    heat[:, :, 1] = 70 * (1 - norm)
    heat[:, :, 2] = 40 * (1 - norm)
    alpha = (norm * mask * 0.78)[:, :, None]
    overlay = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # Draw bounding boxes.
    from PIL import ImageDraw
    draw = ImageDraw.Draw(overlay_img)
    for box in boxes:
        x1 = int(box["x"] * w); y1 = int(box["y"] * h)
        x2 = int((box["x"] + box["width"]) * w); y2 = int((box["y"] + box["height"]) * h)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=max(2, w // 500))

    semantic = None
    if scene_metrics is not None:
        try:
            bm = scene_metrics(before)
            am = scene_metrics(after)
            keys = [
                ("water_like_percent","Water-like"),
                ("vegetation_like_percent","Vegetation-like"),
                ("builtup_like_percent","Built-up-like"),
                ("burn_like_percent","Burn-like")
            ]
            deltas=[]
            for key,label in keys:
                delta=round(am[key]-bm[key],2)
                deltas.append({
                    "metric":label,
                    "before":bm[key],
                    "after":am[key],
                    "delta":delta,
                    "direction":"increased" if delta>0.05 else ("decreased" if delta<-0.05 else "stable")
                })
            semantic={"before_metrics":bm,"after_metrics":am,"deltas":deltas}
        except Exception as exc:
            semantic={"error":str(exc)}

    return {
        "status": "success",
        "method": "pixel-difference + normalized brightness + connected-components",
        "width": w,
        "height": h,
        "changed_pixels": changed_pixels,
        "change_percent": change_percent,
        "regions": len(boxes),
        "sensitivity": sensitivity,
        "threshold": round(threshold, 1),
        "regions_data": boxes,
        "before_preview": data_url(before),
        "after_preview": data_url(after),
        "change_preview": data_url(overlay_img),
        "semantic_changes": semantic,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--sensitivity", type=int, default=35)
    args = p.parse_args()
    print(json.dumps(compare(args.before, args.after, args.sensitivity)))
