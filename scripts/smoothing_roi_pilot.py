"""Geometric skin-region approximation, not an official Instagram algorithm.

Reuse the audited paired-view training/evaluation engine without changing splits.
Smoothing is applied on the native-resolution frozen ROI before resize299.
"""
import json
import sys
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pilgram2_roi_pilot as engine

PROTOCOL = Path(__file__).resolve().parents[1] / "configs/smoothing_roi_pilot_v1.json"

def mask_for(size):
    width, height = size
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x, y = (x + .5) / width, (y + .5) / height
    # Frozen ROI is 1.5x the face bbox. This is geometry, NOT skin parsing.
    radius = np.sqrt(((x-.5)/.29)**2 + ((y-.50)/.35)**2)
    mask = np.clip((1-radius)/.12, 0, 1)
    # Broad eye protection and mouth protection; feather their edges.
    for cx, cy, rx, ry in ((.5,.40,.30,.085), (.5,.65,.19,.075)):
        r = np.sqrt(((x-cx)/rx)**2 + ((y-cy)/ry)**2)
        mask *= np.clip((r-1)/.20, 0, 1)
    return mask.astype(np.float32)

def smooth(image, strength, settings):
    array = np.asarray(image.convert("RGB"))
    # Spatial radius is proportional to ROI size, so native scales are matched.
    spatial = max(1., min(image.size) * settings["sigma_space_roi_fraction"])
    filtered = cv2.bilateralFilter(array, settings["diameter"],
                                  settings["sigma_color_rgb255"], spatial)
    mask = mask_for(image.size)[...,None] * strength
    result = np.rint(array.astype(np.float32)*(1-mask)+filtered.astype(np.float32)*mask)
    return Image.fromarray(np.clip(result,0,255).astype(np.uint8))

def install(protocol):
    original_crop = engine.crop_face_roi
    def tagged(name):
        def transform(image):
            image.info["smoothing_condition"] = name
            return image
        return transform
    for name in protocol["filters"]:
        setattr(engine.pilgram2, name, tagged(name))
    def crop(image, row, fallback_size=299):
        condition = image.info.get("smoothing_condition")
        result = original_crop(image, row, fallback_size=fallback_size)
        if condition:
            if row["roi_status"] != "detected":
                # No reliable face localization: leave fallback unchanged, record count.
                return result
            result = smooth(result, protocol["strengths"][condition], protocol["bilateral"])
        return result
    engine.crop_face_roi = crop

def main():
    if len(sys.argv)>1 and sys.argv[1]=="preview":
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--cache-root',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        args=parser.parse_args(sys.argv[2:])
        frame=engine.pd.read_csv(args.cache_root/'pilot_manifest.csv',dtype=str,keep_default_na=False)
        rows=frame[(frame.split=='train') & (frame.roi_status=='detected')].groupby('method',sort=False).head(1)
        cfg=json.loads((args.cache_root/'config.json').read_text())
        conditions=['normal',*cfg['protocol']['filters']]
        canvas=Image.new('RGB',(299*len(conditions),329*len(rows)),'white')
        draw=ImageDraw.Draw(canvas)
        for iy,row in enumerate(rows.to_dict('records')):
            for ix,condition in enumerate(conditions):
                with Image.open(engine.cache_path(args.cache_root,condition,row)) as opened:
                    canvas.paste(opened.convert('RGB'),(299*ix,329*iy+30))
                draw.text((299*ix+5,329*iy+5),row['method']+' / '+condition,fill='black')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(args.output)
        print('Preview (train images, no test):',args.output,flush=True)
        print('ROI fallback unchanged:',int((frame.roi_status!='detected').sum()),flush=True)
        return
    if "--protocol" not in sys.argv:
        sys.argv.extend(["--protocol", str(PROTOCOL)])
    index = sys.argv.index("--protocol")
    protocol = json.loads(Path(sys.argv[index+1]).read_text())
    # Cache/run identity must reflect actual implementation and OpenCV version.
    protocol["opencv_version"] = cv2.__version__
    protocol["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    install(protocol)
    print("Smoothing pilot: geometric face mask, not Instagram; strength="
          +str(protocol["strengths"]), flush=True)
    engine.main(protocol_override=protocol)

if __name__ == "__main__":
    main()
