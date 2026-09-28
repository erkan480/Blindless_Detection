"""Rebuild the user-facing Colab notebook without notebook dependencies."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip().splitlines(True)})


md("""
# Blindless Detection · APTOS / Colab T4

**Önerilen ilk deney: EfficientNet-B3 + ImageNet, 300 × 300.** Alternatif: EfficientNetV2-S (384) veya B0 (224).

**Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU.** Bu defter TPU için değildir. Hücreleri sırayla çalıştır.
Mevcut denetlenmiş veri ayrımı korunur; test kümesi eğitim/kalibrasyonda kullanılmaz.
Bir araştırma prototipidir; yüksek softmax skoru klinik doğruluk garantisi değildir.
Kod ve manifest paketi kişisel kullanım içindir; APTOS verisini başkalarına dağıtmaz.
""")
md("""## 1 · Ortam kontrolü
Colab'in uyumlu PyTorch/CUDA kurulumunu korur. Eksik küçük bağımlılıklar kurulur; gerçek sürümler kaydedilir.
Yerel geliştirmede PyTorch 2.5.1 / Torchvision 0.20.1 kullanıldı; bu defter aktif Colab sürümünü kısa kontrolden geçirir.
""")
code("""
import sys, os, subprocess, importlib.util
from pathlib import Path
for module, requirement in [("sklearn", "scikit-learn>=1.5,<2"), ("PIL", "Pillow>=10"), ("matplotlib", "matplotlib>=3.8")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", requirement], check=True)
try:
    import torch, torchvision
except Exception as exc:
    raise RuntimeError("Colab PyTorch kurulumu uyumsuz. Temiz bir T4 çalışma zamanı açıp varsayılan PyTorch/Torchvision kurulumu ile yeniden dene.") from exc
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 5), "PyTorch >=2.5 gerekli"
assert torch.cuda.is_available(), "T4 GPU seçili değil. Çalışma zamanı ayarından GPU seç."
print("Python:", sys.version.split()[0], "Torch:", torch.__version__, "Torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0), "Bellek:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
""")
md("""## 2 · Küçük proje paketini yükle
Bilgisayarındaki **colab/blindless_colab_bundle.zip** dosyasını seç. Görüntüler bu pakette yoktur.
Paket içindeki dosyalar SHA-256 ile denetlenir; mevcut Windows yolları taşınmaz.
""")
code("""
from google.colab import files
import zipfile, io, json, hashlib
uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Yalnızca blindless_colab_bundle.zip yükle.")
bundle_bytes = next(iter(uploaded.values()))
PROJECT = Path("/content/BlindlessDetection")
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as z:
    metadata = json.loads(z.read("bundle.json"))
    assert metadata["format"] == "blindless-colab-bundle-v1"
    expected = set(metadata["files"]) | {"bundle.json"}
    assert set(z.namelist()) == expected, "Paket içerik listesi uyuşmuyor"
    assert sum(info.file_size for info in z.infolist()) < 10_000_000, "Beklenmeyen paket boyutu"
    for name, digest in metadata["files"].items():
        destination = (PROJECT / name).resolve()
        if PROJECT.resolve() not in destination.parents:
            raise ValueError("Geçersiz paket yolu")
        content = z.read(name)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Paket dosyası bozuk: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
print("Paket hazır. Bölünmeler:", metadata["splits"])
del uploaded, bundle_bytes
""")
md("""## 3 · Drive'a kalıcı kayıt
Eğitim sonuçları kendi Drive alanına kaydedilir. Oturum koparsa aynı deneme adı ve ayarlarla devam edebilirsin.
GPU oturumunun ve Drive kaydının kesintisiz kalacağı garanti edilmez.
""")
code("""
from google.colab import drive
drive.mount("/content/drive")
RUNS = Path("/content/drive/MyDrive/BlindlessDetection/runs")
RUNS.mkdir(parents=True, exist_ok=True)
""")
md("""## 4 · Drive klasöründeki APTOS verisini kullan
Veri klasörü hazır: **`/content/drive/MyDrive/aptos2019-blindness-detection`**.
Bu hücre `train.csv` ve `train_images` yapısını kontrol eder. Veri ZIP'i veya Kaggle anahtarı gerekmez.
Eğitim, doğrulama ve son değerlendirme için etiketli görüntüler ile mevcut sabit ayrım kullanılır.
""")
code("""
from pathlib import Path
import csv

SOURCE = Path("/content/drive/MyDrive/aptos2019-blindness-detection")
if not SOURCE.is_dir():
    raise FileNotFoundError(
        f"Veri klasörü bulunamadı: {SOURCE}. Önce 3. adımda Drive bağlantısını açın."
    )
train_csv = SOURCE / "train.csv"
train_images = SOURCE / "train_images"
if not train_csv.is_file() or not train_images.is_dir():
    raise FileNotFoundError(
        f"{SOURCE} klasörü doğrudan train.csv dosyasını ve train_images klasörünü içermeli."
    )
with train_csv.open(encoding="utf-8-sig", newline="") as stream:
    reader = csv.DictReader(stream)
    if not {"id_code", "diagnosis"}.issubset(reader.fieldnames or []):
        raise ValueError("train.csv içinde id_code ve diagnosis sütunları bulunmalı.")
    labeled_count = sum(1 for _ in reader)
if labeled_count == 0:
    raise ValueError("train.csv boş; etiketli APTOS verisini kontrol edin.")
print("Veri kaynağı:", SOURCE)
print("Etiketli görüntü kaydı:", labeled_count)
print("Sonraki hazırlık adımı görüntüleri ve etiketleri doğrulayarak yerel önbelleği oluşturacak.")
""")
md("""## 5 · Model ve deneme ayarları
İlk deney B3/300 ile başlar. `efficientnet_v2_s` seçersen 384 piksel ve batch=8 önerilir.
Aynı denemeye devam etmek için ayarları değiştirme. Farklı model/ayar için yeni `RUN_NAME` kullan.
MixUp'u karşılaştırmak istersen ayrı denemede 0 yap; bütün deneyleri doğrulama kümesiyle karşılaştır.
""")
code("""
ARCHITECTURE = "efficientnet_b3"  # efficientnet_b0 / efficientnet_b3 / efficientnet_v2_s
RUN_NAME = "b3_300_seed42_v1"
IMAGE_SIZE = {"efficientnet_b0": 224, "efficientnet_b3": 300, "efficientnet_v2_s": 384}[ARCHITECTURE]
BATCH_SIZE = 12 if ARCHITECTURE != "efficientnet_v2_s" else 8
EPOCHS = 30
MIXUP = 0.1
THRESHOLD = 0.80
if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in (".", ".."):
    raise ValueError("RUN_NAME tek klasör adı olmalı")
RUN = RUNS / RUN_NAME
CACHE = Path("/content/aptos_cache512")
COMMON = ["--architecture", ARCHITECTURE, "--image-size", str(IMAGE_SIZE), "--cache-size", "512",
          "--manifests", str(PROJECT / "data"), "--cache", str(CACHE), "--output", str(RUN),
          "--batch-size", str(BATCH_SIZE), "--epochs", str(EPOCHS), "--mixup", str(MIXUP),
          "--threshold", str(THRESHOLD), "--workers", "2"]
def command(action, extra=(), common=COMMON):
    subprocess.run([sys.executable, "-u", str(PROJECT / "train_colab.py"), action, *common, *extra], cwd=PROJECT, check=True)
RUN.mkdir(parents=True, exist_ok=True)
versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(RUN / "environment.txt").write_text(versions)
print("Model:", ARCHITECTURE, "Girdi:", IMAGE_SIZE, "Deneme:", RUN)
""")
md("""## 6 · Orijinal görüntü doğrulama ve önbellek
Bu adım 3.504 görüntünün orijinal piksel hash'lerini ve etiketlerini doğrular; mevcut 2.452/526/526 ayrımını değiştirmez.
Drive klasöründeki orijinallerden `/content/aptos_cache512` altında 512 piksel önbellek oluşturur. Eğitim bu yerel önbellekten okunur; kaynak görüntüler değiştirilmez. Oturum yenilendiğinde tekrar çalıştır; hazırlık bitmeden eğitime izin verilmez.
""")
code("""
command("prepare", ["--source", str(SOURCE)])
""")
md("""## 7 · Kısa gerçek veri kontrolü
İki küçük eğitim batch'i ve doğrulama çıktısı kontrol edilir. Bu adım rastgele ağırlıklıdır; çıktı gerçek model değildir.
Bellek hatası varsa BATCH_SIZE'ı küçültüp ayar hücresini ve bu hücreyi yeniden çalıştır.
""")
code("""
from datetime import datetime
smoke_common = list(COMMON)
smoke_common[smoke_common.index("--output") + 1] = str(Path("/content") / ("smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
command("smoke", common=smoke_common)
""")
md("""## 8 · Eğitimi başlat / kaldığı yerden devam et
ImageNet ağırlıkları ilk çalıştırmada indirilir. İki epoch ısınma, ardından ince ayar; erken durdurma doğrulama QWK'sına bakar.
Drive'da `last.pt` veya `best.pt` varsa aynı ayarlarla otomatik devam eder. Tamamlanmış bir deneme yeniden eğitilmez.
""")
code("""
resume = ["--resume"] if any((RUN / name).exists() for name in ("last.pt", "best.pt")) else []
command("train", resume)
""")
md("""## 9 · Doğrulama kalibrasyonu ve masaüstü modeli
En iyi epoch seçimi ve sıcaklık kalibrasyonu yalnızca doğrulama kümesine dayanır. Teste dokunulmaz.
""")
code("""
command("export")
validation = json.loads((RUN / "validation_metrics.json").read_text())
print("Doğrulama QWK:", round(validation["qwk"], 4), "Macro F1:", round(validation["macro_f1"], 4))
print("Ağır evre duyarlılığı:", round(validation["classes"]["Severe"]["recall"], 4), "Kabul oranı:", round(validation["coverage"], 4))
import matplotlib.pyplot as plt
history = [json.loads(line) for line in (RUN / "history.jsonl").read_text().splitlines() if line.strip()]
by_epoch = {r["epoch"]: r for r in history}
ordered = [by_epoch[k] for k in sorted(by_epoch)]
plt.figure(figsize=(9, 3))
for key, label in [("qwk", "QWK"), ("macro_f1", "Macro F1")]:
    plt.plot([r["epoch"] for r in ordered], [r[key] for r in ordered], label=label)
plt.xlabel("Epoch"); plt.ylabel("Doğrulama skoru"); plt.legend(); plt.grid(alpha=.2); plt.show()
""")
md("""## 10 · Son test — yalnızca son model seçildikten sonra
Bu hücre varsayılan olarak çalışmaz. B3/V2-S/ayar kararını doğrulama sonuçlarıyla ver, ardından aşağıdaki değeri True yap.
Son test tek kez kaydedilir. B0 geliştirilirken bu test kümesinin sonuçları daha önce görüldü; sonuç **geliştirme karşılaştırmasıdır**, yeni dış/klinik doğrulama değildir.
""")
code("""
FINAL_MODEL_SELECTED = False
if FINAL_MODEL_SELECTED:
    command("test", ["--confirm-final-model"])
else:
    print("Son test atlandı. Model seçimi yalnızca doğrulama kümesinde yapılır.")
""")
md("""## 11 · Modeli indir ve uygulamaya al
Masaüstü uygulamasındaki **Model ekle…** düğmesiyle bu `.pt` dosyasını seç. Mevcut model korunur.
Günlük/checkpoint dosyaları Drive'da kalır. Eski B0 ile aynı görüntüde sonuçları karşılaştırabilirsin; model seçimini tek örneğe göre yapma.
""")
code("""
MODEL = RUN / f"aptos_{ARCHITECTURE}.pt"
if not MODEL.is_file():
    raise FileNotFoundError("Önce eğitim ve dışa aktarma adımlarını tamamla.")
files.download(str(MODEL))
""")
md("""Kaynaklar: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Torchvision](https://docs.pytorch.org/vision/stable/models.html).
Hasta kimliği bulunmadığından hasta düzeyinde ayrım ve yakın kopya ayrımı garanti edilmez. Veri artırma yeni bağımsız hasta verisi sağlamaz.
""")
notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "Blindless_APTOS_T4.ipynb", "provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"blindless-{index:02d}"
path = Path(__file__).with_name("Blindless_APTOS_T4.ipynb")
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
