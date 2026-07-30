# clinical-nlp-specialty-predictor

Klinik transkripsiyon metinlerinden tıbbi uzmanlık alanını tahmin eden bir metin sınıflandırma çalışması.
Veri seti: [mtsamples](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) (4999 rapor, 40 uzmanlık alanı).

Projenin ana bulgusu şu: başarıyı kısıtlayan şey model seçimi ya da hiperparametre değil, **etiket kalitesiydi**.
En büyük kazançlar birbiriyle içerik olarak çakışan sınıfları düzelterek elde edildi.

## Sonuçlar

Nihai kurulum: 6 sınıf, 2726 rapor, %80/20 stratified split (546 test örneği).

| Model | Accuracy | Macro-F1 | Seçilen `C` |
|---|---|---|---|
| **LinearSVC** (SVD + tuned C) | **0.736** | 0.687 | 0.03 |
| **LogisticRegression** (SVD + tuned C) | 0.729 | **0.688** | 0.10 |
| XGBoost (sparse TF-IDF) | 0.632 | 0.570 | — |
| *Taban çizgisi (çoğunluk sınıfı)* | *0.284* | — | — |

### Gelişimin kaynağı

Başlangıç kurulumu (8 sınıf, ayarsız `C`, düz TF-IDF) ile bugünkü durum:

| Model | Başlangıç | Nihai |
|---|---|---|
| LogisticRegression | 0.42 | 0.729 |
| LinearSVC | 0.30 | 0.736 |
| XGBoost | 0.29 | 0.632 |

Katkıların ayrıştırılmış hali (5-fold CV, accuracy):

| Adım | Etki |
|---|---|
| `C` regularizasyon taraması | 0.531 → 0.607 |
| SVD (LSA, 300 bileşen) | 0.607 → 0.616 |
| Surgery etiketinin düzeltilmesi | 0.616 → 0.749 |
| Medikal stop-word listesi | ~0 (macro-F1'e +0.007) |

## Alınan Kararlar

### 1. `Consult - History and Phy.` + `General Medicine` → `General Medical Practice`

İki sınıf da genel anamnez/muayene notu; metin içerikleri ayırt edilemiyordu ve modeller sürekli
birbirine karıştırıyordu. Tek etiket altında birleştirildi.

### 2. Top-8 yerine Top-7

Birleştirme sonrası 8. sıraya `SOAP / Chart / Progress Notes` giriyordu. Bu bir uzmanlık alanı değil
doküman formatı olduğu için tüm sınıflarla çakışıyor — bilerek dışarıda bırakıldı.

### 3. `Surgery` etiketinin dağıtılması

Hata analizinde **tüm hataların %50'si** Surgery kaynaklıydı (recall 0.50; 1088 örneğin 240'ı Orthopedic,
142'si Gastroenterology olarak tahmin ediliyordu). Sebep model değil etiketti: *Surgery* bir organ sistemi
değil, bir işlem türü — ortopedik bir ameliyat notu gerçekten hem Surgery hem Orthopedic.

Surgery satırları `sample_name` + `description` alanındaki işlem adına bakan bir kural tablosuyla organ
sistemine geri etiketlendi:

```
Surgery 1088 → Orthopedic 173 | Cardiovascular 164 | Gastroenterology 144 | Neurology 24
             → 583 satır çıkarıldı
```

Çıkarılanlar kural tablosuna uymayan işlemler (üroloji, oftalmoloji, KBB, jinekoloji — zaten 7 sınıfın
dışındalar) ve omurga cerrahisi gibi gerçekten belirsiz olanlar (`discectomy`, `fusion`: hem ortopedi hem
beyin cerrahisi yapıyor). Yan etki olarak sınıf dağılımı belirgin şekilde dengelendi: 775↔247 (önceden 1088↔223).

### 4. Medikal stop-word listesi

İngilizce stop-word'lere ek olarak ~100 kelimelik bir liste (`surgery`, `patient`, `history`, `procedure`,
`evaluation`, `impression`, `preoperative` …). Bu kelimeler her raporda geçtiği için sınıf ayırt ediciliği yok.
Ölçüldüğünde accuracy katkısı ~0 çıktı; modeli yorumlanabilir kıldığı için tutuldu.

### 5. `C` taraması — projenin en baskın parametresi

5000 özelliğe karşılık ~2200 eğitim örneği olduğu için modeller ezberliyordu. Varsayılan `C=1` ile
LogReg 0.531, `C=0.05` ile 0.607. LinearSVC'de etki daha uçta: `C=1.0` → 0.410, `C=0.005` → 0.604.
Sabit değer yazmak yerine `GridSearchCV` ile train setinde 5-fold CV üzerinden seçiliyor.

### 6. SVD (LSA, 300 bileşen)

Seyrek 5000 boyutlu uzay yerine 300 gizli konsept. Gürültü azalıyor, eşanlamlı terimler (`heart`/`cardiac`)
aynı eksende toplanıyor. Açıklanan varyans 0.589.

### 7. Veri sızıntısı önlemleri

- `keywords` sütunu atıldı — satırların **%97.9'unda** uzmanlık adının kendisini içeriyor.
- `sample_name` ve `description` yalnızca etiket düzeltmesinde kullanıldı, modelleme öncesi atıldı.
  Modele giren tek alan `transcription`.
- TF-IDF ve SVD yalnızca train setinde `fit` edildi.
- Hiperparametre seçimi train setindeki CV ile yapıldı; test seti yalnızca nihai raporlamada kullanıldı.

## Denenip Vazgeçilenler

Aşağıdakiler 5-fold CV ile ölçüldü ve fayda sağlamadığı için kullanılmadı:

| Deneme | Sonuç |
|---|---|
| Ensemble (soft voting: LogReg + ComplementNB + SVD-LogReg) | 0.593 — tek modelden kötü |
| word + char n-gram birleşimi | 0.513 |
| `description` alanını özelliğe eklemek | +0.005 (gürültü içinde) |
| `ngram_range` (1,1) / (1,2) / (1,3) | aralarında anlamlı fark yok |
| `max_features` artırmak | ters etki: 1000 → 0.547, limitsiz → 0.513 |
| `class_weight` kaldırmak | macro-F1 0.474 → 0.338 |
| XGBoost'u SVD özelliklerine geçirmek | 0.634 → 0.637 |

## Bilinen Sınırlamalar

- **Dairesellik:** Surgery etiketleri `description`/`sample_name` anahtar kelimelerinden türetildi ve
  `transcription` da aynı işlem adlarını içeriyor. Bu adımdan sonraki skorlar önceki kurulumla birebir
  kıyaslanabilir değil.
- **Veri kaybı:** Kullanılabilir 4966 rapordan 2726'sı kaldı (%45 elendi). Metriklerin bir kısmı belirsiz
  örneklerin çıkarılmasından geliyor.
- **Tek split:** Tablodaki skorlar tek bir 80/20 bölünmeden. Bu boyutta ±2 puan gürültü var
  (5-fold CV karşılığı: 0.749 ± 0.023).
- **Çözülmemiş çakışma:** `Neurology` (F1 0.47) ve `Radiology` (F1 0.48) hâlâ birbirine karışıyor —
  beyin görüntüleme raporları etikette Radiology ama içerik olarak nöroloji. Aynı türden bir etiket problemi.
- TF-IDF tabanlı bu yaklaşımın tavanı ~0.75 civarında. Daha ileri gitmek için ClinicalBERT /
  Bio_ClinicalBERT embedding'leri gerekiyor.

## Kurulum ve Çalıştırma

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

`mtsamples.csv` dosyasını proje kök dizinine koyup `medical_nlp_classification.ipynb` içinde **Run All**.

## Notebook Yapısı

| Bölüm | İçerik |
|---|---|
| 1. Keşifsel Veri Analizi | Eksik/tekrarlı kayıt kontrolü, sınıf dağılımı |
| 2. Etiket Düzeltmeleri | Consult+General Medicine birleştirmesi, Surgery'nin dağıtılması |
| 3. Metin Temizliği | Bölüm başlıklarının silinmesi, kısaltmaların açılması, regex normalizasyon |
| 4. Özellik Çıkarımı | TF-IDF (medikal stop-word'lerle) + SVD |
| 5. Modeller | LogisticRegression, XGBoost, LinearSVC — her biri `GridSearchCV` ile |
