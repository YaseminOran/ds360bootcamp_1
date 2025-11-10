import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

class SimpleDriftDetector:
    def __init__(self):
        self.reference_data = None
        self.drift_threshold = 0.05  # p-value threshold
        self.psi_threshold_low = 0.1   # PSI düşük drift eşiği
        self.psi_threshold_high = 0.2  # PSI yüksek drift eşiği
        
    def set_reference_data(self, reference_df, target_column='high_risk'):
        """Referans veriyi ayarlar"""
        self.reference_data = reference_df.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in reference_df.columns if col != target_column]
        print(f"Referans veri ayarlandı. Boyut: {reference_df.shape}")
    
    def generate_production_data(self, n_samples=200, drift_factor=0.0):
        """Üretim verisi simülasyonu (drift ile)"""
        if self.reference_data is None:
            raise ValueError("Önce referans veriyi ayarlayın!")
        
        np.random.seed(42)
        production_data = self.reference_data.sample(n=n_samples, replace=True).copy()
        
        if drift_factor > 0:
            print(f"🔄 Drift simülasyonu başlatılıyor (faktör: {drift_factor})")
            
            # Yaş dağılımında belirgin kayma
            age_drift = np.random.normal(drift_factor * 15, drift_factor * 8, n_samples)
            production_data['age'] += age_drift
            print(f"   📊 Yaş ortalama kayması: +{age_drift.mean():.1f} yıl")
            
            # BMI dağılımında kayma
            bmi_drift = np.random.normal(drift_factor * 5, drift_factor * 2, n_samples)
            production_data['bmi'] += bmi_drift
            print(f"   📊 BMI ortalama kayması: +{bmi_drift.mean():.1f}")
            
            # Kan basıncında sistematik artış (daha belirgin)
            bp_drift = drift_factor * 25
            production_data['blood_pressure_systolic'] += bp_drift
            print(f"   📊 Sistolik basınç artışı: +{bp_drift:.1f} mmHg")
            
            # Kolesterol seviyelerinde artış
            chol_drift = np.random.normal(drift_factor * 30, drift_factor * 15, n_samples)
            production_data['cholesterol'] += chol_drift
            print(f"   📊 Kolesterol artışı: +{chol_drift.mean():.1f} mg/dL")
            
            # Sigara içme oranında belirgin değişim
            smoking_change_prob = drift_factor * 0.4  # Daha yüksek değişim oranı
            smoking_changes = np.random.binomial(1, smoking_change_prob, n_samples)
            production_data['smoking'] = np.clip(
                production_data['smoking'] + smoking_changes, 0, 1
            )
            new_smoking_rate = production_data['smoking'].mean()
            old_smoking_rate = self.reference_data['smoking'].mean()
            print(f"   📊 Sigara oranı: {old_smoking_rate:.2f} → {new_smoking_rate:.2f}")
            
            # Egzersiz saatlerinde azalma
            exercise_reduction = np.random.exponential(drift_factor * 2, n_samples)
            production_data['exercise_hours_week'] = np.maximum(
                production_data['exercise_hours_week'] - exercise_reduction, 0
            )
            
            # Stres seviyesinde artış
            stress_increase = np.random.poisson(drift_factor * 2, n_samples)
            production_data['stress_level'] = np.minimum(
                production_data['stress_level'] + stress_increase, 5
            )
        
        # Timestamp ekle
        production_data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            periods=n_samples
        )
        
        return production_data
    
    def kolmogorov_smirnov_test(self, reference_data, current_data, feature):
        """Kolmogorov-Smirnov test ile drift tespiti"""
        ref_values = reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        # KS test
        ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
        
        return {
            'feature': feature,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.drift_threshold
        }
    
    def population_stability_index(self, reference_data, current_data, feature, bins=10):
        """Population Stability Index (PSI) hesaplama"""
        def calculate_psi(ref_values, curr_values, bins):
            # Binleri oluştur
            _, bin_edges = np.histogram(ref_values, bins=bins)
            
            # Her bin için dağılımları hesapla
            ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_values, bins=bin_edges)
            
            # Oranları hesapla
            ref_percents = ref_counts / len(ref_values)
            curr_percents = curr_counts / len(curr_values)
            
            # PSI hesapla
            psi = 0
            for i in range(len(ref_percents)):
                if ref_percents[i] > 0 and curr_percents[i] > 0:
                    psi += (curr_percents[i] - ref_percents[i]) * np.log(curr_percents[i] / ref_percents[i])
            
            return psi
        
        ref_values = reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        psi_value = calculate_psi(ref_values, curr_values, bins)
        
        # PSI yorumlama (daha hassas eşikler)
        if psi_value < self.psi_threshold_low:
            interpretation = "✅ Drift yok"
        elif psi_value < self.psi_threshold_high:
            interpretation = "⚠️ Düşük drift"
        else:
            interpretation = "🚨 Yüksek drift"
        
        return {
            'feature': feature,
            'psi_value': psi_value,
            'interpretation': interpretation,
            'drift_detected': psi_value >= 0.1
        }
    
    def detect_feature_drift(self, current_data):
        """Tüm özellikler için drift tespiti"""
        if self.reference_data is None:
            raise ValueError("Referans veri ayarlanmamış!")
        
        drift_results = []
        
        print("=== FEATURE DRIFT ANALİZİ ===")
        
        for feature in self.feature_columns:
            if feature not in current_data.columns:
                continue
                
            # Sadece numeric features için test yap
            if self.reference_data[feature].dtype in ['int64', 'float64']:
                # KS Test
                ks_result = self.kolmogorov_smirnov_test(
                    self.reference_data, current_data, feature
                )
                
                # PSI Test
                psi_result = self.population_stability_index(
                    self.reference_data, current_data, feature
                )
                
                result = {
                    'feature': feature,
                    'ks_statistic': ks_result['ks_statistic'],
                    'ks_p_value': ks_result['p_value'],
                    'ks_drift': ks_result['drift_detected'],
                    'psi_value': psi_result['psi_value'],
                    'psi_interpretation': psi_result['interpretation'],
                    'psi_drift': psi_result['drift_detected']
                }
                
                drift_results.append(result)
                
                drift_status = "🚨 DRIFT!" if (ks_result['drift_detected'] or psi_result['drift_detected']) else "✅ OK"
                print(f"\n📊 {feature}: {drift_status}")
                print(f"   🔍 KS Test: p={ks_result['p_value']:.4f}")
                print(f"   📈 PSI: {psi_result['psi_value']:.4f} - {psi_result['interpretation']}")
        
        return drift_results
    
    def detect_target_drift(self, current_data):
        """Hedef değişken drift tespiti"""
        if self.target_column not in current_data.columns:
            print("Hedef değişken mevcut değil - target drift analizi yapılamıyor")
            return None
        
        ref_target = self.reference_data[self.target_column]
        curr_target = current_data[self.target_column]
        
        # Oranları karşılaştır
        ref_positive_rate = ref_target.mean()
        curr_positive_rate = curr_target.mean()
        
        rate_change = abs(curr_positive_rate - ref_positive_rate)
        
        # Chi-square test
        from scipy.stats import chi2_contingency
        
        ref_counts = ref_target.value_counts()
        curr_counts = curr_target.value_counts()
        
        # Contingency table oluştur
        contingency_table = pd.DataFrame({
            'reference': ref_counts,
            'current': curr_counts
        }).fillna(0)
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        result = {
            'reference_positive_rate': ref_positive_rate,
            'current_positive_rate': curr_positive_rate,
            'rate_change': rate_change,
            'chi2_statistic': chi2,
            'chi2_p_value': p_value,
            'target_drift_detected': p_value < self.drift_threshold
        }
        
        print("\n=== TARGET DRIFT ANALİZİ ===")
        print(f"Referans pozitif oranı: {ref_positive_rate:.3f}")
        print(f"Güncel pozitif oranı: {curr_positive_rate:.3f}")
        print(f"Oran değişimi: {rate_change:.3f}")
        print(f"Chi-square test p-value: {p_value:.4f}")
        print(f"Target drift: {'EVET' if result['target_drift_detected'] else 'HAYIR'}")
        
        return result
    
    def visualize_feature_drift(self, current_data, feature):
        """Özellik drift görselleştirmesi"""
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            print(f"Özellik {feature} bulunamadı!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram karşılaştırması
        axes[0].hist(self.reference_data[feature], bins=30, alpha=0.7, 
                    label='Referans', color='blue', density=True)
        axes[0].hist(current_data[feature], bins=30, alpha=0.7, 
                    label='Güncel', color='red', density=True)
        axes[0].set_title(f'{feature} - Dağılım Karşılaştırması')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Yoğunluk')
        axes[0].legend()
        
        # Box plot karşılaştırması
        data_to_plot = [self.reference_data[feature], current_data[feature]]
        axes[1].boxplot(data_to_plot, labels=['Referans', 'Güncel'])
        axes[1].set_title(f'{feature} - Box Plot Karşılaştırması')
        axes[1].set_ylabel(feature)
        
        plt.tight_layout()
        plt.show()
    
    def create_drift_dashboard(self, drift_results):
        """Drift dashboard oluşturur"""
        if not drift_results:
            print("Drift sonucu bulunmuyor")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # PSI skorları
        features = [r['feature'] for r in drift_results]
        psi_values = [r['psi_value'] for r in drift_results]
        
        axes[0, 0].bar(features, psi_values, color=['red' if r['psi_drift'] else 'green' for r in drift_results])
        axes[0, 0].axhline(y=0.1, color='orange', linestyle='--', label='Düşük Drift Eşiği')
        axes[0, 0].axhline(y=0.2, color='red', linestyle='--', label='Yüksek Drift Eşiği')
        axes[0, 0].set_title('Population Stability Index (PSI)')
        axes[0, 0].set_ylabel('PSI Değeri')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        
        # KS Test p-values
        ks_p_values = [r['ks_p_value'] for r in drift_results]
        
        axes[0, 1].bar(features, ks_p_values, color=['red' if r['ks_drift'] else 'green' for r in drift_results])
        axes[0, 1].axhline(y=0.05, color='red', linestyle='--', label='Anlamlılık Eşiği (0.05)')
        axes[0, 1].set_title('Kolmogorov-Smirnov Test P-Values')
        axes[0, 1].set_ylabel('P-Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        
        # Drift özeti
        total_features = len(drift_results)
        psi_drifted = sum(1 for r in drift_results if r['psi_drift'])
        ks_drifted = sum(1 for r in drift_results if r['ks_drift'])
        
        drift_summary = ['PSI Drift', 'KS Drift']
        drift_counts = [psi_drifted, ks_drifted]
        
        axes[1, 0].bar(drift_summary, drift_counts, color=['orange', 'red'])
        axes[1, 0].set_title('Drift Tespiti Özeti')
        axes[1, 0].set_ylabel('Drift Tespit Edilen Özellik Sayısı')
        
        # Genel durum
        overall_status = "DRIFT TESPİT EDİLDİ" if (psi_drifted > 0 or ks_drifted > 0) else "DRIFT YOK"
        status_color = "red" if (psi_drifted > 0 or ks_drifted > 0) else "green"
        
        axes[1, 1].text(0.5, 0.5, overall_status, 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=16, color=status_color, weight='bold',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Genel Durum')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def simulate_drift_monitoring():
    """Drift izleme sistemini simüle eder"""
    print("=== SIMPLE MODEL DRIFT İZLEME SİMÜLASYONU ===")
    
    # Risk skorlama sisteminden veri al
    from risk_scoring import HealthRiskScoring
    
    risk_system = HealthRiskScoring()
    df = risk_system.generate_sample_data(1000)
    X_train, X_test, y_train, y_test = risk_system.prepare_data(df)
    risk_system.train_models(X_train, y_train)
    
    # Drift detector'ı başlat
    drift_detector = SimpleDriftDetector()
    
    # Referans veri olarak eğitim setini kullan
    train_df = X_train.copy()
    train_df['high_risk'] = y_train
    drift_detector.set_reference_data(train_df)
    
    # Farklı drift seviyelerinde test (daha belirgin seviyeler)
    drift_levels = [0.0, 0.3, 0.6, 1.0]
    
    for i, drift_level in enumerate(drift_levels):
        print(f"\n{'='*50}")
        print(f"DRIFT SEVİYESİ: {drift_level}")
        print('='*50)
        
        # Üretim verisi oluştur
        production_data = drift_detector.generate_production_data(
            n_samples=200, 
            drift_factor=drift_level
        )
        
        # Feature drift tespiti
        drift_results = drift_detector.detect_feature_drift(production_data)
        
        # Target drift tespiti
        target_drift = drift_detector.detect_target_drift(production_data)
        
        # Görselleştirmeler
        if drift_level > 0:
            print(f"\nDrift görselleştirmeleri oluşturuluyor...")
            drift_detector.visualize_feature_drift(production_data, 'age')
            drift_detector.visualize_feature_drift(production_data, 'bmi')
        
        # Dashboard
        drift_detector.create_drift_dashboard(drift_results)
        
        # Öneriler
        any_drift = any(r['psi_drift'] or r['ks_drift'] for r in drift_results)
        if any_drift or (target_drift and target_drift['target_drift_detected']):
            print("\n🚨 UYARI: Drift tespit edildi!")
            print("  - Model performansını kontrol edin")
            print("  - Veri kaynağını analiz edin")
            print("  - Model yeniden eğitimi değerlendirin")
        else:
            print("\n✅ Drift seviyesi kabul edilebilir")

def main():
    """Ana fonksiyon"""
    print("=== SIMPLE DRIFT DETECTION SYSTEM ===")
    
    # Drift izleme simülasyonunu çalıştır
    simulate_drift_monitoring()
    
    print("\n=== ÖNERİLER ===")
    print("1. PSI > 0.1: Düşük drift, izleme devam edin")
    print("2. PSI > 0.2: Yüksek drift, model yeniden eğitimi gerekli")
    print("3. KS test p < 0.05: İstatistiksel olarak anlamlı drift")
    print("4. Target drift: Hedef değişken dağılımında değişiklik")
    print("5. Düzenli monitoring ve otomatik uyarı sistemleri kurun")

if __name__ == "__main__":
    main()