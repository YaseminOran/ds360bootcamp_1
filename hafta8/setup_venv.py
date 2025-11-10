#!/usr/bin/env python3
"""
Hafta 8 - Sağlık Risk Skorlama projesi için otomatik kurulum scripti
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Komut çalıştır ve sonucu göster"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} başarılı!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} başarısız!")
        print(f"Hata: {e.stderr}")
        return False

def main():
    """Ana kurulum fonksiyonu"""
    print("=" * 50)
    print("HAFTA 8 - SAĞLIK RİSK SKORLAMA")
    print("Otomatik Kurulum Başlatılıyor...")
    print("=" * 50)
    
    # Proje dizinini kontrol et
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"📁 Çalışma dizini: {project_dir}")
    
    # Python versiyonunu kontrol et
    python_version = sys.version_info
    print(f"🐍 Python versiyonu: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Uyarı: Python 3.8+ önerilir")
    
    # Virtual environment oluştur
    venv_path = project_dir / "venv"
    
    if venv_path.exists():
        print("📦 Virtual environment zaten mevcut")
        response = input("Yeniden oluşturmak istiyor musunuz? (y/N): ")
        if response.lower() == 'y':
            print("🗑️  Eski virtual environment siliniyor...")
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("✅ Mevcut virtual environment kullanılacak")
    
    if not venv_path.exists():
        success = run_command(
            f"{sys.executable} -m venv venv", 
            "Virtual environment oluşturuluyor"
        )
        if not success:
            print("❌ Virtual environment oluşturulamadı!")
            return False
    
    # Aktivasyon komutlarını belirle
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    print(f"📝 Aktivasyon komutu: {activate_cmd}")
    
    # pip'i güncelle
    success = run_command(
        f"{pip_cmd} install --upgrade pip",
        "pip güncelleniyor"
    )
    if not success:
        print("⚠️  pip güncellemesi başarısız, devam ediliyor...")
    
    # Requirements.txt'yi yükle
    requirements_file = project_dir / "requirements.txt"
    if requirements_file.exists():
        success = run_command(
            f"{pip_cmd} install -r requirements.txt",
            "Gerekli paketler yükleniyor"
        )
        if not success:
            print("❌ Paket yükleme başarısız!")
            return False
    else:
        print("❌ requirements.txt dosyası bulunamadı!")
        return False
    
    # Data dizini oluştur
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"📁 Data dizini oluşturuldu: {data_dir}")
    
    # Test çalıştırması
    test_script = """
import sys
sys.path.append('src')

try:
    from risk_scoring import HealthRiskScoring
    from drift_detection import ModelDriftDetector
    print("✅ Tüm modüller başarıyla import edildi!")
    
    # Basit test
    risk_system = HealthRiskScoring()
    df = risk_system.generate_sample_data(100)
    print(f"✅ Test verisi oluşturuldu: {df.shape}")
    
    print("🎉 Kurulum testi başarılı!")
    
except Exception as e:
    print(f"❌ Import hatası: {e}")
    sys.exit(1)
"""
    
    print("\n🧪 Kurulum testi yapılıyor...")
    test_file = project_dir / "test_installation.py"
    test_file.write_text(test_script)
    
    success = run_command(
        f"{python_cmd} test_installation.py",
        "Kurulum testi"
    )
    
    # Test dosyasını temizle
    test_file.unlink()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 KURULUM TAMAMLANDI!")
        print("=" * 50)
        print("\n📋 Sonraki adımlar:")
        print(f"1. Virtual environment'ı aktif edin:")
        print(f"   {activate_cmd}")
        print("\n2. Ana scripti çalıştırın:")
        print("   python src/risk_scoring.py")
        print("\n3. Drift tespiti için:")
        print("   python src/drift_detection.py")
        print("\n4. Jupyter notebook'ları kullanın:")
        print("   jupyter notebook notebooks/risk_scoring_example.ipynb")
        
        print("\n📚 Proje yapısı:")
        print("├── src/")
        print("│   ├── risk_scoring.py      # Ana risk skorlama modülü")
        print("│   └── drift_detection.py   # Model drift tespiti")
        print("├── notebooks/")
        print("│   └── risk_scoring_example.ipynb")
        print("├── data/                    # Veri ve raporlar")
        print("├── requirements.txt")
        print("└── README.md")
        
    else:
        print("\n❌ Kurulum başarısız!")
        print("Lütfen hata mesajlarını kontrol edin ve tekrar deneyin.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)