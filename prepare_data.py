"""Audit original APTOS labels and create fixed, deduplicated train/val/test splits."""
import argparse, csv, hashlib, json
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from sklearn.model_selection import train_test_split
from src.ai.preprocessing import ROOT, crop_retina, PREPROCESS

def prepare_one(row, source, cache):
    ident=row['id_code'];label=int(row['diagnosis'])
    if label not in range(5):raise ValueError(f'Invalid label: {row}')
    path=source/'train_images'/f'{ident}.png'
    with Image.open(path) as img:
        img.load();rgb=img.convert('RGB')
        digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        target=cache/f'{ident}.png'
        # Regenerate cache to avoid trusting files from another data source.
        crop_retina(rgb).resize((320,320),Image.Resampling.LANCZOS).save(target)
    return {'id_code':ident,'diagnosis':label,'sha256':digest,'path':str(target)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    args=p.parse_args()
    out=ROOT/'data'; out.mkdir(exist_ok=True)
    cache=out/'cache320'; cache.mkdir(exist_ok=True)
    with (args.data/'train.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    records=[]; seen={}; duplicates=[]; conflicts=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,record in enumerate(pool.map(lambda row:prepare_one(row,args.data,cache),rows)):
            digest=record['sha256']
            if digest in seen:
                duplicates.append(record['id_code'])
                if seen[digest]['diagnosis'] != record['diagnosis']: conflicts.append(digest)
            else:
                seen[digest]=record;records.append(record)
            if (i+1)%100==0:print(f'Audit {i+1}/{len(rows)}',flush=True)
    # Conflicting exact duplicates are excluded entirely, not assigned an arbitrary label.
    records=[r for r in records if r['sha256'] not in set(conflicts)]
    train,rest=train_test_split(records,test_size=.3,stratify=[r['diagnosis'] for r in records],random_state=42)
    val,test=train_test_split(rest,test_size=.5,stratify=[r['diagnosis'] for r in rest],random_state=42)
    summary={'source':str(args.data.resolve()),'preprocess':PREPROCESS,'source_rows':len(rows),
             'exact_duplicates_removed':len(duplicates),'conflicting_hashes':len(set(conflicts)),
             'seed':42,'limitations':'No patient identifiers: patient-level separation and near-duplicate exclusion are not guaranteed.'}
    for name,part in [('train',train),('val',val),('test',test)]:
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id_code','diagnosis','sha256','path']);w.writeheader();w.writerows(part)
        summary[name]={'count':len(part),'classes':dict(Counter(r['diagnosis'] for r in part))}
    assert not ({r['sha256'] for r in train}&{r['sha256'] for r in val})
    assert not ({r['sha256'] for r in test}&{r['sha256'] for r in train+val})
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
