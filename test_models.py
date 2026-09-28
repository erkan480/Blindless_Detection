"""Checkpoint compatibility and import checks; no pretrained download needed."""
import csv
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.ai.retina import (CLASSES, MODEL_REGISTRY, PREPROCESS, ROOT,
                          RetinaPredictor, build_model, discover_models, import_model)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.checkpoint = cls.folder / 'fixture.pt'
        cls.checkpoint.touch()
        cls.weights = build_model().state_dict()
        cls.state = {'state_dict': cls.weights, 'classes': CLASSES,
                     'preprocess': PREPROCESS, 'image_size': 224,
                     'architecture': 'efficientnet_b0', 'selected_epoch': 3}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load_state(self, state):
        with patch('src.ai.retina.torch.load', return_value=state) as loader:
            result = RetinaPredictor(self.checkpoint)
            loader.assert_called_once_with(self.checkpoint.resolve(), map_location='cpu', weights_only=True)
            return result

    def test_supported_models_produce_five_scores_without_downloads(self):
        with patch('torch.hub.download_url_to_file', side_effect=AssertionError('Unexpected download')):
            for architecture in MODEL_REGISTRY:
                with self.subTest(architecture=architecture):
                    model = build_model(architecture=architecture).eval()
                    with torch.inference_mode():
                        output = model(torch.zeros(1, 3, 128, 128))
                    self.assertEqual(tuple(output.shape), (1, 5))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertEqual(model.classifier[0].p, .3)
                    del model

    def test_unsupported_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model(architecture='xception')

    def test_legacy_checkpoint_and_normalized_metadata(self):
        legacy = {key: value for key, value in self.state.items() if key != 'architecture'}
        predictor = self.load_state(legacy)
        self.assertEqual(predictor.architecture, 'efficientnet_b0')
        self.assertEqual(predictor.display_name, 'EfficientNet-B0')
        self.assertEqual(predictor.cache_size, 320)
        self.assertEqual(predictor.metadata['selected_epoch'], 3)
        self.assertEqual(predictor.metadata['architecture'], 'efficientnet_b0')
        self.assertNotIn('state_dict', predictor.metadata)
        self.assertEqual(predictor.checkpoint_path, self.checkpoint.resolve())

    def test_malformed_metadata_is_rejected_before_model_construction(self):
        malformed = [
            ('smoke_only', True),
            ('architecture', 'unknown'), ('architecture', []),
            ('classes', CLASSES[::-1]), ('preprocess', 'different-v2'),
            ('image_size', 127), ('image_size', 513), ('image_size', True),
            ('image_size', 224.5), ('image_size', None),
            ('cache_size', 0), ('cache_size', float('inf')),
            ('temperature', 0), ('temperature', -1), ('temperature', float('nan')),
            ('temperature', float('inf')), ('temperature', '1.0'),
            ('threshold', -.1), ('threshold', 1.1), ('threshold', float('nan')),
            ('threshold', True), ('state_dict', {}), ('state_dict', None),
        ]
        for key, value in malformed:
            with self.subTest(key=key, value=value):
                with patch('src.ai.retina.build_model') as constructor:
                    with self.assertRaises(ValueError):
                        self.load_state({**self.state, key: value})
                    constructor.assert_not_called()
        with self.assertRaises(ValueError):
            self.load_state([])

    def test_state_dict_requires_exact_architecture(self):
        incomplete = dict(self.weights)
        incomplete.pop(next(iter(incomplete)))
        with self.assertRaisesRegex(ValueError, 'mimariyle'):
            self.load_state({**self.state, 'state_dict': incomplete})

    def test_nonfinite_weights_are_rejected(self):
        invalid = dict(self.weights)
        invalid['classifier.1.bias'] = torch.full((5,), float('nan'))
        with self.assertRaisesRegex(ValueError, 'sayısal'):
            self.load_state({**self.state, 'state_dict': invalid})

    def test_discovery_does_not_load_models(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name in ('z.pt', 'a.PT', 'notes.txt'):
                (folder / name).touch()
            (folder / 'directory.pt').mkdir()
            with patch('src.ai.retina.torch.load', side_effect=AssertionError('Discovery must not load')):
                self.assertEqual([p.name for p in discover_models(folder)], ['a.PT', 'z.pt'])
                self.assertEqual(discover_models(folder / 'absent'), [])

    def test_import_validates_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / 'source' / 'model.pt'
            source.parent.mkdir()
            torch.save(self.state, source)
            library = folder / 'models'
            library.mkdir()
            existing = library / 'model.pt'
            existing.write_bytes(b'keep this original')
            installed = import_model(source, library)
            self.assertEqual(installed.name, 'model_2.pt')
            self.assertEqual(existing.read_bytes(), b'keep this original')
            self.assertEqual(installed.read_bytes(), source.read_bytes())
            self.assertEqual(import_model(installed, library), installed)
            bad = folder / 'invalid.pt'
            bad.write_bytes(b'invalid checkpoint')
            untouched = folder / 'untouched'
            with self.assertRaises(ValueError):
                import_model(bad, untouched)
            self.assertFalse(untouched.exists())

    def test_actual_trained_checkpoint_prediction(self):
        checkpoint = ROOT / 'models' / 'aptos_efficientnet_b0.pt'
        if not checkpoint.is_file():
            self.skipTest('Existing local trained model is unavailable')
        predictor = RetinaPredictor(checkpoint)
        with (ROOT / 'data' / 'test.csv').open(newline='', encoding='utf-8') as stream:
            row = next(csv.DictReader(stream))
        result = predictor.predict(row['path'])
        self.assertEqual(len(result['scores']), 5)
        self.assertAlmostEqual(sum(result['scores']), 1., places=5)
        self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in result['scores']))
        self.assertEqual(result['score'], max(result['scores']))
        self.assertEqual(result['class_id'], result['top_class'] if result['score'] >= predictor.threshold else -1)


if __name__ == '__main__':
    unittest.main()
