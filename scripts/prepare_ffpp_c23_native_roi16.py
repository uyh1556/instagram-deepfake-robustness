"""Stream unmodified c23 video frames into the existing Celeb-DF native ROI policy.

No full-frame PNG dataset, new split, JPEG encoding, or source deletion.
Each completed video is resumable. Unknown source IDs remain excluded.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import create_frozen_face_roi_manifest as roi
from xception_preprocessing import crop_face_roi


def atomic_json(path,value):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=['original','Deepfakes','Face2Face','FaceSwap','NeuralTextures'],required=True)
    p.add_argument('--anchor-manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    anchor={};anchor_bytes=args.anchor_manifest.read_bytes()
    for row in csv.DictReader(anchor_bytes.decode().splitlines()):
        for identity in row['source_ids'].split('|'):
            if identity in anchor and anchor[identity]!=row['split']:raise ValueError('Anchor source-ID leakage')
            anchor[identity]=row['split']
    roi.tf.config.threading.set_intra_op_parallelism_threads(2)
    roi.tf.config.threading.set_inter_op_parallelism_threads(2)
    cv2.setNumThreads(1)
    args.output_root.mkdir(parents=True,exist_ok=True)
    detector=roi.MTCNN(device=roi.select_device('cpu'))
    start=time.monotonic()
    for method in args.methods:
        source=args.input_root/('original_sequences/youtube' if method=='original' else f'manipulated_sequences/{method}')/'c23/videos'
        videos=sorted(source.glob('*.mp4'))
        if args.limit:videos=videos[:args.limit]
        if not videos or (not args.limit and len(videos)!=1000):raise ValueError(f'{method}: expected1000 completed videos, found{len(videos)}')
        meta=args.output_root/'metadata'/method;meta.mkdir(parents=True,exist_ok=True)
        config={'name':'ffpp_c23_native_roi16_v1','method':method,'source_root':str(source.resolve()),
                'frames_per_video':16,'sampling':'unique rounded linspace(0,frame_count-1,min(16,frame_count))',
                'roi':'same Celeb-DF MTCNN filters, max detection side1024, expansion1.5, RGB128 padding',
                'output':'native square lossless PNG; fallback letterbox299; no aligned256 or JPEG',
                'split':'reuse existing source IDs; unknown/cross-split sources excluded',
                'anchor_sha256':hashlib.sha256(anchor_bytes).hexdigest(),
                'versions':{k:importlib.metadata.version(k) for k in ['numpy','pillow','mtcnn','tensorflow']},
                'roi_code_sha256':roi.sha256_file(Path(roi.__file__))}
        path=meta/'config.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Existing method protocol differs')
        if not path.exists():atomic_json(path,config)
        records=[]
        for ordinal,video in enumerate(videos,1):
            signature={'bytes':video.stat().st_size,'mtime_ns':video.stat().st_mtime_ns}
            done=meta/f'{video.stem}.json'
            if done.exists():
                old=json.loads(done.read_text())
                if old['source_signature']!=signature:raise ValueError('Source video changed')
                if not all((args.output_root/r['crop_path']).is_file() for r in old['frames']):raise FileNotFoundError('Completed crop missing')
                records.extend(old['frames']);continue
            if shutil.disk_usage(args.output_root).free<4*2**30:raise RuntimeError('Free disk below4GiB; sources preserved; resume after freeing space')
            cap=cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():raise RuntimeError(f'Cannot open {video}')
                count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
                if count<1 or fps<=0:raise ValueError('Invalid video metadata')
                indices=np.unique(np.rint(np.linspace(0,count-1,min(16,count))).astype(int))
                ids=video.stem.split('_');roles={anchor.get(i,'excluded_unknown_source') for i in ids}
                split=next(iter(roles)) if len(roles)==1 else 'excluded_cross_source'
                rows=[]
                for offset in range(0,len(indices),8):
                    images=[];dimages=[];selected=indices[offset:offset+8]
                    for idx in selected:
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)):raise RuntimeError('Seek failed')
                        ok,bgr=cap.read()
                        if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=idx:raise RuntimeError('Frame decode/seek mismatch')
                        image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                        images.append(image);dimages.append(roi.resize_for_detection(image,1024))
                    detected=detector.detect_faces([np.asarray(x,dtype=np.uint8) for x in dimages],fit_to_image=True,box_format='xywh',output_type='json')
                    if len(detected)!=len(images):raise RuntimeError('Detector batch length mismatch')
                    for idx,image,dimage,detection in zip(selected,images,dimages,detected):
                        detection_row=roi.create_detection_row(int(idx),image.size,dimage.size,detection,1.5)
                        cropped=crop_face_roi(image,detection_row)
                        rel=f'images/{method}/{video.stem}/{idx:06d}.png'
                        dest=args.output_root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                        temp=dest.with_suffix('.tmp.png');cropped.save(temp,format='PNG',compress_level=6)
                        if dest.exists():
                            if roi.sha256_file(dest)!=roi.sha256_file(temp):raise ValueError('Existing partial crop differs')
                            temp.unlink()
                        else:temp.replace(dest)
                        rows.append({'crop_path':rel,'video_id':method+'/'+video.stem,'source_video':str(video.resolve()),
                                     'source_ids':'|'.join(ids),'split':split,'method':method,
                                     'label':'real' if method=='original' else 'fake','class_index':int(method!='original'),
                                     'frame_index':int(idx),'timestamp_seconds':int(idx)/fps,
                                     'crop_width':cropped.width,'crop_height':cropped.height,'png_bytes':dest.stat().st_size,**detection_row})
                atomic_json(done,{'source_signature':signature,'frames':rows});records.extend(rows)
            finally:cap.release()
            if ordinal%10==0 or ordinal==len(videos):
                print(f'{method}: {ordinal}/{len(videos)} videos | {len(records)} crops | elapsed{(time.monotonic()-start)/60:.1f}min | free{shutil.disk_usage(args.output_root).free/2**30:.1f}GiB',flush=True)
                if ordinal==10:
                    mean_bytes=sum(r['png_bytes'] for r in records)/len(records)
                    estimate=mean_bytes*16*(len(videos)-ordinal)
                    if shutil.disk_usage(args.output_root).free<estimate*1.3+4*2**30:raise RuntimeError('Estimated crops exceed free disk safety margin; partial outputs preserved')
        blob_path=meta/'crop_manifest.csv';temp=blob_path.with_suffix('.tmp.csv')
        with temp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        temp.replace(blob_path)
        atomic_json(meta/'summary.json',{'complete':True,'videos':len(videos),'images':len(records),
            'fallback':sum(r['roi_status']=='fallback_letterbox' for r in records),'png_bytes':sum(r['png_bytes'] for r in records)})
        print('Completed:',method,flush=True)


if __name__=='__main__':main()
