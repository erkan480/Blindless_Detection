"""Reproducible two-stage EfficientNet-B0 fine tuning; test only after selection."""
import argparse, csv, json, random, time, os, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
from src.ai.retina import ROOT, CLASSES, PREPROCESS, build_model, transform

class Retinas(Dataset):
    def __init__(self, split, training=False, size=224):
        with (ROOT/'data'/f'{split}.csv').open() as f: self.rows=list(csv.DictReader(f))
        self.transform=transform(size,training)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(row['path']) as img: x=self.transform(img.convert('RGB'))
        return x,int(row['diagnosis'])

def metrics(y,logits):
    pred=logits.argmax(1)
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'qwk':float(cohen_kappa_score(y,pred,weights='quadratic',labels=list(range(5)))),
            'confusion_matrix':confusion_matrix(y,pred,labels=list(range(5))).tolist(),
            'classes':classification_report(y,pred,labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)}

def atomic_save(state,path):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)

def evaluate(model,loader,device):
    model.eval(); logits=[];labels=[]
    with torch.inference_mode():
        for x,y in loader:
            logits.append(model(x.to(device)).cpu());labels.append(y)
    return torch.cat(labels),torch.cat(logits)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.smoke: raise RuntimeError('GPU unavailable; full training was not started.')
    out=ROOT/'runs'/('smoke' if a.smoke else 'efficientnet_b0');out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() and not a.resume and not a.smoke:
        raise RuntimeError('Existing run found. Use --resume to preserve it.')
    datasets={s:Retinas(s,s=='train',a.image_size) for s in ['train','val','test']}
    split_hash=hashlib.sha256(b''.join((ROOT/'data'/f'{s}.csv').read_bytes() for s in ['train','val','test'])).hexdigest()
    loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=s=='train',num_workers=a.workers,
             pin_memory=device.type=='cuda',persistent_workers=a.workers>0) for s,d in datasets.items()}
    model=build_model(pretrained=not a.resume).to(device)
    counts=np.bincount([int(r['diagnosis']) for r in datasets['train'].rows],minlength=5)
    weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device=device);weights/=weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW([{'params':model.features.parameters(),'lr':1e-4},
                                {'params':model.classifier.parameters(),'lr':1e-3}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start=0;best=-float('inf');stale=0
    if a.resume:
        ck=torch.load(out/'last.pt',map_location=device,weights_only=True)
        if ck.get('split_hash')!=split_hash or ck['image_size']!=a.image_size:
            raise ValueError('Resume requires the original data splits and image size.')
        model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        start=ck['epoch']+1;best=ck['best'];stale=ck['stale']
    print(f'Device={device}; model=EfficientNet-B0; batch={a.batch_size}; splits='+str({s:len(d) for s,d in datasets.items()}),flush=True)
    config={**vars(a),'device':str(device),'torch':str(torch.__version__),'initialization':'ImageNet1K','seed':42}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for epoch in range(start,1 if a.smoke else a.epochs):
        t=time.time();warmup=epoch<2 and not a.smoke
        for par in model.features.parameters(): par.requires_grad=not warmup
        model.train()
        # Freeze BN statistics for stability on small batches and during warmup.
        for module in model.features.modules():
            if isinstance(module,nn.BatchNorm2d): module.eval()
        total=0.;n=0
        for step,(x,y) in enumerate(loaders['train']):
            x=x.to(device);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=loss_fn(model(x),y)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();total+=float(loss)*len(y);n+=len(y)
            if step%50==0: print(f'Epoch {epoch+1} batch {step+1}/{len(loaders["train"])} loss={float(loss):.4f}',flush=True)
            if a.smoke and step>=2: break
        if a.smoke:
            x,y=next(iter(loaders['val']));model.eval()
            with torch.inference_mode(): pred=model(x.to(device))
            assert pred.shape==(len(y),5) and torch.isfinite(pred).all()
            print('SMOKE PASSED: forward/backward and validation',flush=True);return
        y,logits=evaluate(model,loaders['val'],device);m=metrics(y.numpy(),logits.numpy())
        improved=m['qwk']>best
        if improved:best=m['qwk'];stale=0
        else:stale+=1
        if stale and stale%2==0:
            for g in optimizer.param_groups:g['lr']*=.5
        state={'state_dict':model.state_dict(),'classes':CLASSES,'preprocess':PREPROCESS,'image_size':a.image_size,
               'temperature':1.,'threshold':.8,'epoch':epoch,'best':best,'stale':stale,'split_hash':split_hash,
               'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()}
        atomic_save(state,out/'last.pt')
        if improved:atomic_save(state,out/'best.pt')
        event={'epoch':epoch+1,'loss':total/n,'seconds':time.time()-t,**m}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(f'Epoch {epoch+1}: val QWK={m["qwk"]:.4f}, macroF1={m["macro_f1"]:.4f}, time={time.time()-t:.0f}s',flush=True)
        if stale>=5 and epoch>=7:break
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
    yv,lv=evaluate(model,loaders['val'],device)
    temperatures=torch.linspace(.5,3.,101)
    temperature=float(min(temperatures,key=lambda t:float(nn.functional.cross_entropy(lv/t,yv))))
    yt,lt=evaluate(model,loaders['test'],device)
    report=metrics(yt.numpy(),lt.numpy());probs=(lt/temperature).softmax(1);scores,pred=probs.max(1);accepted=scores>=.8
    baseline=np.zeros((len(yt),5));baseline[:,int(counts.argmax())]=1
    report['majority_baseline']=metrics(yt.numpy(),baseline)
    report.update(temperature=temperature,threshold=.8,threshold_policy='Fixed exploratory threshold; not a clinical guarantee',
                  coverage=float(accepted.float().mean()),accepted_accuracy=float((pred[accepted]==yt[accepted]).float().mean()) if accepted.any() else None,
                  note='Held-out subset of labeled APTOS train.csv; competition test.csv has no ground-truth labels.')
    (out/'test_metrics.json').write_text(json.dumps(report,indent=2))
    with (out/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id_code','true','predicted','score','accepted'])
        for r,y,prediction,score,ok in zip(datasets['test'].rows,yt,pred,scores,accepted):w.writerow([r['id_code'],int(y),int(prediction),float(score),bool(ok)])
    target=ROOT/'models';target.mkdir(exist_ok=True)
    export={k:ck[k] for k in ['state_dict','classes','preprocess','image_size']};export.update(temperature=temperature,threshold=.8)
    export.update(architecture='efficientnet_b0',initialization='ImageNet1K',split_hash=split_hash,selected_epoch=ck['epoch']+1)
    atomic_save(export,target/'aptos_efficientnet_b0.pt')
    print('TRAINING COMPLETE '+json.dumps(report),flush=True)

if __name__=='__main__':main()
