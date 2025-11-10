# Hafta 8 - Sağlık Sprinti: Risk Skorlama

Bu proje sağlık verilerini kullanarak risk skorlama sisteminin geliştirilmesini kapsamaktadır.

## Ana Konular

### Risk Skoru Hesaplama
- **Logistic Regression** ile risk skoru hesaplama
- **Random Forest** ile risk skoru hesaplama
- Model karşılaştırması ve performans analizi

### Model Drift Tespiti
- **Evidently AI** ile model drift tespiti
- Model performansının zaman içindeki değişiminin izlenmesi
- Otomatik uyarı sistemleri

## Kurulum

1. Virtual environment oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

1. `src/risk_scoring.py` - Risk skorlama modelleri
2. `src/drift_detection.py` - Model drift tespiti
3. `notebooks/` - Jupyter notebook örnekleri

## Veri Seti

Proje sentetik sağlık verileri kullanmaktadır. Gerçek hasta bilgileri içermez.