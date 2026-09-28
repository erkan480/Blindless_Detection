# Colab T4 ile APTOS modeli eğitimi

Başlangıç adayı **EfficientNet-B3, 300 × 300**. İkinci deney olarak **EfficientNetV2-S, 384 × 384** seçilebilir. B0 küçük ve hızlı bir karşılaştırma seçeneğidir. Tüm gerçek eğitimler ImageNet ağırlıklarıyla başlar. Daha büyük modelin daha iyi sonuç vereceği garanti değildir.

## Başlatma

1. `colab/Blindless_APTOS_T4.ipynb` dosyasını [Google Colab](https://colab.research.google.com/) üzerinden yükle. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU** seç. Bu defter CUDA GPU içindir; TPU desteklemez.
2. Hücreleri sırayla çalıştır. İlk yükleme adımında `colab/blindless_colab_bundle.zip` dosyasını seç. Küçük paket yalnızca kodu ve mevcut veri bölünmelerini içerir; görüntüler ve API anahtarları içermez. Proje kodu değişirse paketi `.venv-train\Scripts\python.exe build_colab_bundle.py` ile yeniden oluştur.
3. Google Drive bağlantısını onayla. Checkpoint, günlük ve model dosyaları kişisel Drive alanındaki `BlindlessDetection/runs/` klasörüne yazılır. Drive bağlantısına bu nedenle ihtiyaç vardır; kayıt kesintilere karşı yine de garanti değildir.
4. Veri hücresi doğrudan **`/content/drive/MyDrive/aptos2019-blindness-detection`** klasörünü kullanır. Bu klasör `train.csv` ve `train_images/` içermeli. Görüntüler zaten açılmış olduğundan veri ZIP’i yüklemek veya Kaggle anahtarı eklemek gerekmez. Hazırlık, Drive’daki kaynakları okuyup `/content/aptos_cache512` altında yerel önbellek oluşturur; eğitim bu önbellekten yürür. Kaynak görüntüler değiştirilmez. İkinci adımdaki küçük ZIP yalnızca proje kodu ve sabit veri ayrımıdır.
5. Model ayarını seç, veri doğrulama ve kısa kontrol hücrelerini çalıştır, ardından eğitimi başlat. B3 için batch=12; bellek yetersizse ilk eğitimden önce 8 veya 4'e indir. V2-S için batch=8 ile başla. İlk kaydı olan bir denemede parametre değiştireceksen yeni `RUN_NAME` seç.
6. Eğitimden sonra dışa aktarma hücresi yalnızca doğrulama kümesinde sıcaklık kalibrasyonu yapar. `validation_metrics.json` ve grafiklerden deneyleri karşılaştır. Model seçimini doğrulama QWK, macro F1 ve özellikle **ağır evre duyarlılığı** ile değerlendir.
7. Son model kararını verdikten sonra isteğe bağlı son test hücresinde `FINAL_MODEL_SELECTED = True` yap. Teste göre yeniden model/eşik ayarlama. B0 geliştirilirken aynı test kümesi daha önce görüldü; bu nedenle sonuçlar bağımsız yeni klinik doğrulama olarak sunulamaz.
8. `aptos_efficientnet_b3.pt` (veya seçtiğin mimari) dosyasını indir. Masaüstü uygulamasındaki **Model ekle…** düğmesiyle seç. `.h5`/TensorFlow dosyaları bu PyTorch model akışıyla uyumlu değildir. Mevcut B0 modeli korunur.

## Eğitimde neler değişti?

- Mevcut denetlenmiş **2.452 / 526 / 526** eğitim/doğrulama/test ayrımı korunur. Orijinal görüntülerin piksel SHA-256 değeri ve etiketi doğrulanır; bilgisayara özgü yollar Colab'e taşınmaz.
- Sadece eğitimde döndürme, yatay/dikey çevirme, küçük ölçek, parlaklık/kontrast değişiklikleri ve isteğe bağlı hafif MixUp uygulanır. Yeni görüntü dosyaları çoğaltılmaz; her epoch farklı örnekler üretilir. Çoğaltma yeni bağımsız hasta verisi sağlamaz.
- Sınıf dengesizliği karekök ters frekanslı kayıpla ele alınır; ayrıca oversampling uygulanmaz.
- İki epoch sınıflandırıcı ısınması, ardından düşük öğrenme oranıyla ince ayar; AdamW, karma hassasiyet, gradyan sınırlama, doğrulama QWK'ya göre erken durdurma ve öğrenme oranı düşürme kullanılır.
- Aynı retina kırpma/normalleştirme sözleşmesi korunur. Yeni modeller için orijinallerden **512 × 512 önbellek** oluşturulur; boyut model dosyasına kaydedilir ve masaüstü aynı işlemi uygular. Eski B0 320 önbelleğini kullanmaya devam eder. B3/V2-S yükseltilmiş 320 önbellekten eğitilmez.
- Doğrulamada sıcaklık kalibrasyonu; sabit %80 kabul eşiği. Skorlar tıbbi güven veya doğruluk olasılığı değildir. Kabul edilen oran ve kabul edilen örneklerin başarısı ayrı raporlanır.

## Kesintiden devam

Yeni Colab oturumunda GPU/Drive/kod/veri hücrelerini yeniden çalıştır. Aynı `RUN_NAME` ve eğitim ayarlarını seç. Eğitim hücresi `last.pt` veya `best.pt` varsa otomatik `--resume` ekler. En yeni tam epoch kaydından devam edilir; yarım kalan epoch yeniden çalışır. Veri/mimari/boyut/hiperparametre uyuşmazlığında devam reddedilir. Rastgele sayı üreteçleri ve optimizer durumu saklanır; farklı GPU/runtime sürümlerinde bit düzeyinde aynı sonuç garanti edilmez.

`history.jsonl`: epoch ölçümleri ve sınıf bazında raporlar. `last.pt`: devam kaydı. `best.pt`: doğrulama QWK'ya göre en iyi kayıt. `validation_metrics.json`: seçim ve kalibrasyon. `test_metrics.json`: yalnızca açık son test adımından sonra oluşur. Bunlardan `aptos_*.pt` masaüstüne alınacak dosyadır.

## Ek araştırma öncelikleri

Öncelik ağır evrenin kaçırılmasını azaltmak olmalı. Yakın kopya ve etiket incelemesi, hasta kimliği varsa hasta bazlı bölme, farklı merkezden bağımsız doğrulama ve sınıf bazlı güven aralıkları izleyen geliştirmelerdir. Mevcut veri yalnızca birebir piksel kopyalarından arındırılmıştır; hasta düzeyi veya yakın kopya ayrımı garanti edilmez. Önce basit B3 deneyini değerlendir; MixUp ve mimari karşılaştırmalarını doğrulama üzerinde kontrollü ayrı deneyler olarak yap.

Kaynaklar: [Torchvision modelleri](https://docs.pytorch.org/vision/stable/models.html), [Colab kaynak sınırları](https://research.google.com/colaboratory/faq.html). Colab donanım erişimi ve oturum süresi garanti edilmez. Bu akış yerelde kısa gerçek veri kontrolünden geçirilir; senin Colab hesabında tam eğitim henüz çalıştırılmamıştır.
