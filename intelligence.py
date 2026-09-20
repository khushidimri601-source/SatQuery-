"""Higher-level SatQuery analytics: quality, multi-image timeline and report generation."""
import json, os, base64, io
from PIL import Image
import numpy as np
from ai_model_bridge import run, load_image, metrics
from compare_images import compare
try:
    import multispectral_analysis as ms
except Exception:
    ms = None

def quality(path):
    if ms is not None and ms.is_geotiff(path):
        try:
            data=ms.analyze(path,"generic")
            m=data["summary"]
            warnings=[]
            if m.get("valid_pixel_percent",100)<90: warnings.append("Significant nodata / invalid pixels")
            return {"score":round(max(0,min(100,m.get("valid_pixel_percent",100))),1),"usable":m.get("valid_pixel_percent",0)>=70,"metrics":m,"warnings":warnings,"input_type":"multispectral"}
        except Exception as exc:
            return {"score":0,"usable":False,"metrics":{},"warnings":[f"Multispectral read failed: {exc}"]}
    img=load_image(path); m=metrics(img)
    arr=np.asarray(img).astype(np.float32)/255
    clipped=float(((arr<.02)|(arr>.98)).mean()*100)
    score=max(0,min(100,100-clipped*1.8-abs(m['brightness_percent']-50)*.45))
    return {"score":round(score,1),"usable":score>=55,"metrics":m,"warnings":(["High clipping / possible exposure issue"] if clipped>8 else [])+(["Very dark or bright scene"] if m['brightness_percent']<18 or m['brightness_percent']>88 else []),"input_type":"rgb"}

def timeline(paths,sensitivity=35):
    items=[]
    for i,p in enumerate(paths):
        if ms is not None and ms.is_geotiff(p):
            try: m=ms.analyze(p,"generic")["summary"]
            except Exception: m={}
        else:
            m=metrics(load_image(p))
        items.append({
            "index":i+1,
            "name":os.path.basename(p),
            "quality":quality(p),
            "metrics":m
        })
    changes=[]
    for i in range(len(paths)-1):
        r=compare(paths[i],paths[i+1],sensitivity)
        sem=r.get("semantic_changes") or {}
        changes.append({
            "from":items[i]["name"],
            "to":items[i+1]["name"],
            "change_percent":r["change_percent"],
            "regions":r["regions"],
            "regions_data":r["regions_data"],
            "semantic_changes":sem.get("deltas",[]),
            "change_preview":r.get("change_preview")
        })
    if not changes:
        trend = "Single-scene baseline recorded (add another scene for multi-date change trends)" if len(items) == 1 else "insufficient data"
    else:
        first=changes[0]["change_percent"]; last=changes[-1]["change_percent"]
        if last-first>1: trend="change activity increasing"
        elif first-last>1: trend="change activity decreasing"
        else: trend="stable / mixed change activity"
    return {"status":"success","images":items,"changes":changes,"trend":trend}

def report(query,analysis=None,comparison=None,timeline_data=None):
    from datetime import datetime
    lines=[
        "SATQUERY AI",
        "REMOTE-SENSING INTELLIGENCE REPORT",
        "="*64,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Analyst query: {query or 'General scene intelligence'}"
    ]
    if analysis:
        m=analysis.get('image_metrics') or {}
        lines += [
            "", "1. EXECUTIVE SUMMARY", "-"*64,
            analysis.get('answer') or "Scene analysis completed.",
            "", "2. SCENE METRICS", "-"*64,
            f"Image dimensions          : {m.get('width','—')} × {m.get('height','—')} px",
            f"Water-like area           : {m.get('water_like_percent','—')}%",
            f"Vegetation-like area      : {m.get('vegetation_like_percent','—')}%",
            f"Built-up-like area        : {m.get('builtup_like_percent','—')}%",
            f"Burn-like area            : {m.get('burn_like_percent','—')}%",
            f"Brightness                : {m.get('brightness_percent','—')}%",
            f"Contrast                  : {m.get('contrast_percent','—')}%",
            f"Candidate regions         : {analysis.get('feature_count',0)}",
            f"Analysis engine           : {analysis.get('model_source','CV screening')}"
        ]
    if comparison:
        lines += ["", "3. BEFORE / AFTER CHANGE DETECTION", "-"*64,
                  f"Changed area             : {comparison.get('change_percent','—')}%",
                  f"Detected change regions  : {comparison.get('regions','—')}",
                  f"Detection method         : {comparison.get('method','pixel difference + normalization')}"]
        if 'ndvi' in comparison.get('deltas',{}):
            lines += [f"Flood spectral signal    : {comparison.get('flood_signal_percent','—')}%",
                      f"Vegetation-loss signal   : {comparison.get('vegetation_loss_percent','—')}%",
                      f"Burn spectral signal     : {comparison.get('burn_signal_percent','—')}%",
                      f"Mean NDVI delta          : {comparison['deltas'].get('ndvi','—')}",
                      f"Mean NDWI delta          : {comparison['deltas'].get('ndwi','—')}",
                      f"Mean NBR delta           : {comparison['deltas'].get('nbr','—')}" ]
        sem=(comparison.get("semantic_changes") or {}).get("deltas",[])
        if sem:
            lines += ["", "Textual change evidence:"]
            for d in sem:
                sign="+" if d["delta"]>0 else ""
                lines.append(f"- {d['metric']}: {d['before']}% → {d['after']}% ({sign}{d['delta']} pp; {d['direction']})")
    if timeline_data:
        lines += ["", "4. TEMPORAL ANALYSIS", "-"*64,
                  f"Scenes analysed           : {len(timeline_data.get('images',[]))}",
                  f"Trend                     : {timeline_data.get('trend','—')}"]
        for c in timeline_data.get("changes",[]):
            lines.append(f"- {c['from']} → {c['to']}: {c['change_percent']}% changed area across {c['regions']} region(s)")
            for d in c.get("semantic_changes",[]):
                sign="+" if d["delta"]>0 else ""
                lines.append(f"  • {d['metric']}: {d['before']}% → {d['after']}% ({sign}{d['delta']} pp)")
    lines += [
        "", "5. INTERPRETATION & LIMITATIONS", "-"*64,
        "RGB processing provides deterministic visual screening. Six-band GeoTIFF mode adds NDVI/NDWI/NBR evidence.",
        "Prithvi-EO-2.0 is a pretrained EO backbone; it is not presented as a task-specific detector without",
        "a compatible downstream checkpoint. Flood, vegetation loss and burn signals remain evidence, not ground truth.",
        "High-stakes decisions should use co-registered imagery, cloud/nodata screening and labelled validation data.",
        "", "6. RECOMMENDED NEXT STEP", "-"*64,
        "Use Before/After for event assessment or Timeline for multi-date monitoring. "
        "For operational accuracy, provide calibrated GeoTIFF multispectral/SAR data and labelled ground truth."
    ]
    return "\n".join(lines)


if __name__ == '__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--quality')
    ap.add_argument('--report')
    ap.add_argument('--timeline', nargs='*')
    ap.add_argument('--sensitivity', type=int, default=35)
    a=ap.parse_args()
    if a.quality:
        print(json.dumps(quality(a.quality)))
    elif a.report:
        with open(a.report,'r',encoding='utf-8') as f: d=json.load(f)
        print(json.dumps({'status':'success','report':report(d.get('query',''),d.get('analysis'),d.get('comparison'),d.get('timeline'))}))
    elif a.timeline:
        print(json.dumps(timeline(a.timeline,a.sensitivity)))
