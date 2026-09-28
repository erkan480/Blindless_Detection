"""Focused checks for preprocessing parity, split leakage, and checkpoint inference."""
import csv, tempfile, unittest, os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from src.ai.retina import ROOT, CLASSES, PREPROCESS, crop_retina, transform, build_model, RetinaPredictor

class PipelineTests(unittest.TestCase):
    def test_ui_uncertain_result(self):
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from PyQt5.QtWidgets import QApplication
        from src.ui.view.main_screen import MainScreen
        app=QApplication.instance() or QApplication([])
        window=MainScreen()
        window.show_result({'class_id':-1,'top_class':2,'scores':[.1,.1,.6,.1,.1]})
        self.assertIn('belirsiz',window.result_title.text())
        self.assertEqual(window.bars[2].value(),60)
        window.close()

    def test_crop_and_inference_cache_parity(self):
        a=np.zeros((140,220,3),dtype=np.uint8);a[10:130,30:190]=[150,60,20]
        img=Image.fromarray(a)
        cropped=crop_retina(img)
        self.assertEqual(cropped.width,cropped.height)
        cache=cropped.resize((320,320),Image.Resampling.LANCZOS)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p=Path(tmp)/'image.png';cache.save(p)
            with Image.open(p) as saved:
                self.assertTrue(torch.equal(transform()(saved),transform()(cache)))
        self.assertTrue(torch.isfinite(transform()(crop_retina(Image.new('RGB',(20,30))))).all())

    def test_checkpoint_and_threshold(self):
        model=build_model();model.eval()
        with torch.no_grad():
            model.classifier[1].weight.zero_();model.classifier[1].bias.zero_()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            ck=Path(tmp)/'model.pt';image=Path(tmp)/'image.png'
            Image.new('RGB',(160,160),'red').save(image)
            torch.save({'state_dict':model.state_dict(),'classes':CLASSES,'preprocess':PREPROCESS,
                        'image_size':224,'temperature':1.,'threshold':.8},ck)
            result=RetinaPredictor(ck).predict(image)
            self.assertEqual(result['class_id'],-1)
            self.assertAlmostEqual(sum(result['scores']),1.,places=5)

    def test_split_disjointness(self):
        parts=[]
        for split in ['train','val','test']:
            with (ROOT/'data'/f'{split}.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(set(int(r['diagnosis']) for r in rows),set(range(5)))
            self.assertTrue(all(Path(r['path']).is_file() for r in rows))
            parts.append(set(r['sha256'] for r in rows))
        for i in range(3):
            for j in range(i):self.assertFalse(parts[i]&parts[j])

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
