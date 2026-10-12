"""Resumable, full-resolution Celeb-DF V2 PNG extraction (no face filtering)."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import shutil
import struct
import time
from pathlib import Path
import cv2
import numpy as np

GROUPS = {'Celeb-real':'real', 'YouTube-real':'real', 'Celeb-synthesis':'fake'}

def atomic_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--frames-per-video',type=int,default=16)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    assert args.frames_per_video>0 and args.workers>0
    test_file=args.video_root/'List_of_testing_videos.txt'
    tests={}
    for line in test_file.read_text().splitlines():
        if not line.strip(): continue
        flag,path=line.split(maxsplit=1)
        assert path not in tests and (args.video_root/path).is_file()
        assert int(flag)==(1 if GROUPS[path.split('/')[0]]=='real' else 0)
        tests[path]=int(flag)
    videos=sorted(p for group in GROUPS for p in (args.video_root/group).glob('*.mp4'))
    assert len(videos)==6529 and len(tests)==518, 'Unexpected V2 inventory'
    config={'dataset':'Celeb-DF-v2','video_root':str(args.video_root.resolve()),
            'frames_per_video':args.frames_per_video,'format':'PNG lossless compression6',
            'sampling':'unique rounded linspace(0, frame_count-1, min(16,frame_count))',
            'resize':None,'face_filter':None,'opencv_version':cv2.__version__,
            'official_test_list_sha256':hashlib.sha256(test_file.read_bytes()).hexdigest(),
            'video_inventory_sha256':hashlib.sha256('\n'.join(f'{p.relative_to(args.video_root)}:{p.stat().st_size}' for p in videos).encode()).hexdigest(),
            'labels':{'real':0,'fake':1},'split_policy':'official_test vs non_test_pool; NO new train/val split'}
    args.output_root.mkdir(parents=True,exist_ok=True)
    cfg=args.output_root/'config.json'
    if cfg.exists(): assert json.loads(cfg.read_text())==config, 'Existing output protocol differs'
    else: atomic_json(cfg,config)
    (args.output_root/'official_test_list.txt').write_bytes(test_file.read_bytes())
    cv2.setNumThreads(1)
    def one(video):
        relative=video.relative_to(args.video_root)
        folder=args.output_root/'frames'/relative.with_suffix('')
        done=folder/'frames.json'
        signature={'size':video.stat().st_size,'mtime_ns':video.stat().st_mtime_ns}
        if done.exists():
            data=json.loads(done.read_text())
            assert data['source_signature']==signature
            if all((args.output_root/r['source_path']).is_file() and (args.output_root/r['source_path']).stat().st_size==r['png_bytes'] for r in data['frames']):
                return data
        if shutil.disk_usage(args.output_root).free<3*2**30:
            raise RuntimeError('Free disk below 3 GiB: stop and resume after freeing space')
        cap=cv2.VideoCapture(str(video))
        try:
            assert cap.isOpened(), 'Cannot open video'
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
            assert count>0 and fps>0, 'Invalid video metadata'
            indices=np.unique(np.rint(np.linspace(0,count-1,min(args.frames_per_video,count))).astype(int))
            folder.mkdir(parents=True,exist_ok=True)
            rows=[]
            for index in indices:
                assert cap.set(cv2.CAP_PROP_POS_FRAMES,int(index)), 'Cannot seek'
                ok,frame=cap.read()
                if not ok: raise RuntimeError(f'Cannot decode frame {index}')
                # Position should point to the frame after the decoded frame.
                actual=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
                assert actual==int(index), f'Seek mismatch: requested {index}, got {actual}'
                path=folder/f'{index:06d}.png'
                ok,encoded=cv2.imencode('.png',frame,[cv2.IMWRITE_PNG_COMPRESSION,6])
                assert ok
                blob=encoded.tobytes()
                if path.exists():
                    assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(blob).digest(), 'Existing frame differs'
                else:
                    temp=path.with_suffix('.png.tmp'); temp.write_bytes(blob); temp.replace(path)
                rows.append({'source_path':str(path.relative_to(args.output_root)),
                             'video_path':str(relative),'video_id':str(relative.with_suffix('')),
                             'method':relative.parts[0],'label':GROUPS[relative.parts[0]],
                             'class_index':0 if GROUPS[relative.parts[0]]=='real' else 1,
                             'split':'test' if str(relative) in tests else 'non_test_pool',
                             'frame_index':int(index),'timestamp_seconds':int(index)/fps,
                             'width':frame.shape[1],'height':frame.shape[0],
                             'png_bytes':len(blob),'png_sha256':hashlib.sha256(blob).hexdigest()})
            data={'source_signature':signature,'video_path':str(relative),'total_video_frames':count,
                  'fps':fps,'frames':rows}
            atomic_json(done,data)
            return data
        finally: cap.release()
    results=[]; errors=[]; start=time.monotonic(); last=start
    print(f'Extracting {len(videos)} videos x {args.frames_per_video}; workers={args.workers}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,p):p for p in videos}
        for finished,future in enumerate(concurrent.futures.as_completed(futures),1):
            video=futures[future]
            try: results.append(future.result())
            except Exception as exc:
                errors.append({'video':str(video.relative_to(args.video_root)),'error':str(exc)})
                print('ERROR',errors[-1],flush=True)
            now=time.monotonic()
            if now-last>=30 or finished==len(videos):
                print(f'{finished}/{len(videos)} videos | errors={len(errors)} | elapsed={(now-start)/60:.1f}min | free={shutil.disk_usage(args.output_root).free/2**30:.1f}GiB',flush=True)
                last=now
    rows=sorted((r for d in results for r in d['frames']),key=lambda r:r['source_path'])
    if rows:
        temp=args.output_root/'manifest.csv.tmp'
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        temp.replace(args.output_root/'manifest.csv')
    counts={group:sum(r['method']==group for r in rows) for group in GROUPS}
    summary={'complete':not errors,'videos':len(videos),'successful_videos':len(results),
             'images':len(rows),'group_image_counts':counts,
             'test_images':sum(r['split']=='test' for r in rows),
             'non_test_pool_images':sum(r['split']=='non_test_pool' for r in rows),
             'png_bytes':sum(r['png_bytes'] for r in rows),'errors':errors}
    atomic_json(args.output_root/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if errors: raise SystemExit(1)

if __name__=='__main__': main()
