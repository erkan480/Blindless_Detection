"""Shared checkpoint contract for local and Colab retinal-image models.

Only supported torchvision architectures and tensor-only ``.pt`` checkpoints
are accepted. A checkpoint is validated before it can replace the active model.
"""
from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path
import shutil
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
PREPROCESS = 'retina-border-crop-rgb-imagenet-v1'
MODEL_REGISTRY = {
    'efficientnet_b0': {'display_name': 'EfficientNet-B0', 'image_size': 224},
    'efficientnet_b3': {'display_name': 'EfficientNet-B3', 'image_size': 300},
    'efficientnet_v2_s': {'display_name': 'EfficientNetV2-S', 'image_size': 384},
}
SUPPORTED_ARCHITECTURES = tuple(MODEL_REGISTRY)

from src.ai.preprocessing import crop_retina

def transform(size=224, training=False):
    ops = [transforms.Resize((size, size))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=.12, contrast=.12)]
    return transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])])

def build_model(pretrained=False, architecture='efficientnet_b0'):
    """Build a supported 5-class model; inference never downloads weights."""
    if not isinstance(architecture, str) or architecture not in MODEL_REGISTRY:
        raise ValueError(f'Desteklenmeyen model mimarisi: {architecture}')
    if pretrained:
        torch.hub.set_dir(str(ROOT / '.cache' / 'torch'))
    constructors = {
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
        'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    }
    constructor, weights = constructors[architecture]
    model = constructor(weights=weights if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(model.classifier[1].in_features, 5))
    return model


def _integer_metadata(state, name, default=None, lower=128, upper=512):
    value = state.get(name, default)
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'Model {name} değeri {lower}–{upper} aralığında bir tam sayı olmalı.')
    return value


def _float_metadata(state, name, default, lower, upper, exclusive_lower=False):
    value = state.get(name, default)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'Model {name} değeri geçerli bir sayı olmalı.')
    value = float(value)
    if (not math.isfinite(value) or value > upper
            or (value <= lower if exclusive_lower else value < lower)):
        raise ValueError(f'Model {name} değeri geçerli aralıkta değil.')
    return value


class RetinaPredictor:
    def __init__(self, checkpoint=None):
        path = Path(checkpoint) if checkpoint else ROOT/'models'/'aptos_efficientnet_b0.pt'
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Eğitilmiş model bulunamadı: {path}')
        if path.suffix.lower() != '.pt':
            raise ValueError('Desteklenen model dosyası biçimi .pt olmalı.')
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise ValueError('Model dosyası güvenli biçimde okunamadı. Uyumlu bir .pt paketi seçin.') from exc
        if not isinstance(state, Mapping):
            raise ValueError('Model dosyası ağırlıkları ve eğitim bilgilerini içeren bir paket olmalı.')
        if state.get('smoke_only', False) is not False:
            raise ValueError('Kısa kontrol (smoke) çıktısı eğitilmiş bir model değildir. Eğitim sonrası dışa aktarılan aptos_*.pt dosyasını seçin.')
        if state.get('preprocess') != PREPROCESS or state.get('classes') != CLASSES:
            raise ValueError('Model sınıfları veya ön işleme sürümü uyuşmuyor.')
        architecture = state.get('architecture', 'efficientnet_b0')
        if not isinstance(architecture, str) or architecture not in MODEL_REGISTRY:
            raise ValueError(f'Desteklenmeyen model mimarisi: {architecture}')
        self.architecture = architecture
        self.image_size = _integer_metadata(state, 'image_size')
        self.cache_size = _integer_metadata(state, 'cache_size', 320, upper=2048)
        self.threshold = _float_metadata(state, 'threshold', .8, 0., 1.)
        self.temperature = _float_metadata(state, 'temperature', 1., 0., math.inf, exclusive_lower=True)
        weights = state.get('state_dict')
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError('Model dosyasında geçerli state_dict ağırlıkları bulunamadı.')
        self.model = build_model(architecture=architecture)
        try:
            self.model.load_state_dict(weights, strict=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError('Model ağırlıkları, bildirilen mimariyle veya beş sınıflı çıkışla uyuşmuyor.') from exc
        if any(not torch.isfinite(tensor).all().item() for tensor in self.model.state_dict().values()
               if tensor.is_floating_point()):
            raise ValueError('Model ağırlıklarında geçersiz sayısal değerler var.')
        self.model.eval()
        self.transform = transform(self.image_size)
        self.checkpoint_path = path
        self.display_name = MODEL_REGISTRY[architecture]['display_name']
        self.metadata = {key: value for key, value in state.items() if key != 'state_dict'}
        self.metadata.update(architecture=architecture, image_size=self.image_size,
                             cache_size=self.cache_size, threshold=self.threshold,
                             temperature=self.temperature)

    def predict(self, path):
        with Image.open(path) as image:
            prepared = crop_retina(image).resize((self.cache_size, self.cache_size), Image.Resampling.LANCZOS)
            batch = self.transform(prepared).unsqueeze(0)
        with torch.inference_mode():
            probs = (self.model(batch)/self.temperature).softmax(1)[0]
        if not torch.isfinite(probs).all():
            raise RuntimeError('Model geçerli bir skor üretemedi.')
        score, label = probs.max(0)
        return {'class_id': int(label) if float(score) >= self.threshold else -1,
                'top_class': int(label), 'score': float(score), 'scores': probs.tolist()}


def discover_models(directory=None):
    """List local .pt paths without loading their weights or trusting metadata."""
    folder = Path(directory) if directory is not None else ROOT / 'models'
    if not folder.is_dir():
        return []
    return sorted((p.resolve() for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.pt'),
                  key=lambda p: p.name.casefold())


def import_model(source, directory=None):
    """Validate and copy a checkpoint into the model library without overwriting.

    Returns its installed Path. Existing names get ``_2``, ``_3``, etc. A model
    already in the requested library is validated and its existing path returned.
    The GUI can load an external checkpoint directly with RetinaPredictor(path)
    when it does not need a permanent library copy.
    """
    source = Path(source).expanduser().resolve()
    predictor = RetinaPredictor(source)
    del predictor
    folder = (Path(directory) if directory is not None else ROOT / 'models').expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if source.parent == folder:
        return source
    suffix = 1
    while True:
        name = source.name if suffix == 1 else f'{source.stem}_{suffix}{source.suffix}'
        destination = folder / name
        try:
            output = destination.open('xb')
        except FileExistsError:
            suffix += 1
            continue
        try:
            with source.open('rb') as input_file, output:
                shutil.copyfileobj(input_file, output)
        except Exception:
            output.close()
            destination.unlink(missing_ok=True)
            raise
        return destination
