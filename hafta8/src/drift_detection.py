import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift, TestDataDrift
import pickle
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class ModelDriftDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.reference_data = None
        self.column_mapping = None
        self.drift_reports = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Eğitilmiş modeli yükler"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model yüklendi: {model_path}")
    
    def set_reference_data(self, reference_df, target_column='high_risk'):
        """Referans veriyi ayarlar (eğitim verisi)"""
        self.reference_data = reference_df.copy()
        
        # Evidently için kolon haritalama
        feature_columns = [col for col in reference_df.columns if col != target_column]
        
        self.column_mapping = ColumnMapping(
            target=target_column,
            prediction=None,
            numerical_features=[col for col in feature_columns 
                              if reference_df[col].dtype in ['int64', 'float64']],
            categorical_features=[col for col in feature_columns 
                                if reference_df[col].dtype == 'object' or 
                                reference_df[col].nunique() <= 10]
        )
        
        print(f"Referans veri ayarlandı. Boyut: {reference_df.shape}")
    
    def generate_production_data(self, n_samples=200, drift_factor=0.0):
        """Üretim verisi simülasyonu (drift ile)"""
        if self.reference_data is None:
            raise ValueError("Önce referans veriyi ayarlayın!")
        
        np.random.seed(42)
        production_data = self.reference_data.sample(n=n_samples, replace=True).copy()
        
        if drift_factor > 0:
            # Yaş dağılımında kayma
            age_drift = np.random.normal(0, drift_factor * 10, n_samples)
            production_data['age'] += age_drift
            
            # BMI dağılımında kayma
            bmi_drift = np.random.normal(0, drift_factor * 3, n_samples)
            production_data['bmi'] += bmi_drift
            
            # Kan basıncında sistematik artış
            bp_drift = drift_factor * 15
            production_data['blood_pressure_systolic'] += bp_drift
            
            # Sigara içme oranında değişim
            smoking_change = np.random.binomial(1, drift_factor * 0.2, n_samples)
            production_data['smoking'] = np.clip(
                production_data['smoking'] + smoking_change, 0, 1
            )
        
        # Timestamp ekle
        production_data['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            periods=n_samples
        )
        
        return production_data
    
    def detect_data_drift(self, current_data, save_report=True):
        """Veri kaymasını tespit eder"""
        if self.reference_data is None:
            raise ValueError("Referans veri ayarlanmamış!")
        
        # Drift raporu oluştur
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Raporu kaydet
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"../data/drift_report_{timestamp}.html"
            data_drift_report.save_html(report_path)
            print(f"Drift raporu kaydedildi: {report_path}")
        
        return data_drift_report
    
    def detect_target_drift(self, current_data, save_report=True):
        """Hedef değişken kaymasını tespit eder"""
        if self.reference_data is None:
            raise ValueError("Referans veri ayarlanmamış!")
        
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"../data/target_drift_report_{timestamp}.html"
            target_drift_report.save_html(report_path)
            print(f"Hedef drift raporu kaydedildi: {report_path}")
        
        return target_drift_report
    
    def run_drift_tests(self, current_data):
        """Drift testleri çalıştırır"""
        if self.reference_data is None:
            raise ValueError("Referans veri ayarlanmamış!")
        
        test_suite = TestSuite(tests=[
            TestDataDrift(),
            TestColumnDrift(column='age'),
            TestColumnDrift(column='bmi'),
            TestColumnDrift(column='blood_pressure_systolic'),
            TestColumnDrift(column='cholesterol'),
        ])
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Test sonuçlarını al
        results = test_suite.as_dict()
        
        print("=== DRIFT TEST SONUÇLARI ===")
        for test in results['tests']:
            test_name = test['name']
            status = test['status']
            print(f"{test_name}: {status}")
        
        return test_suite
    
    def monitor_model_performance(self, current_data, model_predictions):
        """Model performansını izler"""
        if 'high_risk' not in current_data.columns:
            print("Gerçek etiketler mevcut değil - performans izleme yapılamıyor")
            return None
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = current_data['high_risk']
        y_pred = model_predictions
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'timestamp': datetime.now()
        }
        
        print("=== MODEL PERFORMANS METRİKLERİ ===")
        for metric, value in metrics.items():
            if metric != 'timestamp':
                print(f"{metric}: {value:.3f}")
        
        return metrics
    
    def create_drift_dashboard(self, drift_history):
        """Drift izleme dashboard'u oluşturur"""
        if not drift_history:
            print("Drift geçmişi bulunmuyor")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Drift skorları zaman serisi
        timestamps = [entry['timestamp'] for entry in drift_history]
        drift_scores = [entry['drift_score'] for entry in drift_history]
        
        axes[0, 0].plot(timestamps, drift_scores, marker='o')
        axes[0, 0].set_title('Veri Drift Skoru - Zaman')
        axes[0, 0].set_ylabel('Drift Skoru')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Feature drift dağılımı
        feature_drifts = {}
        for entry in drift_history:
            for feature, score in entry['feature_drifts'].items():
                if feature not in feature_drifts:
                    feature_drifts[feature] = []
                feature_drifts[feature].append(score)
        
        features = list(feature_drifts.keys())
        avg_drifts = [np.mean(feature_drifts[f]) for f in features]
        
        axes[0, 1].bar(features, avg_drifts)
        axes[0, 1].set_title('Ortalama Feature Drift Skorları')
        axes[0, 1].set_ylabel('Drift Skoru')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Model performans trendi
        if 'model_performance' in drift_history[0]:
            performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for i, metric in enumerate(performance_metrics):
                values = [entry['model_performance'][metric] for entry in drift_history 
                         if 'model_performance' in entry]
                axes[1, i//2].plot(timestamps[:len(values)], values, 
                                 marker='o', label=metric)
        
        axes[1, 0].set_title('Model Performans Trendi')
        axes[1, 0].legend()
        axes[1, 1].set_title('Drift Uyarı Durumu')
        
        plt.tight_layout()
        plt.show()

def simulate_drift_monitoring():
    """Drift izleme sistemini simüle eder"""
    print("=== MODEL DRIFT İZLEME SİMÜLASYONU ===")
    
    # Risk skorlama sisteminden veri al
    from risk_scoring import HealthRiskScoring
    
    risk_system = HealthRiskScoring()
    df = risk_system.generate_sample_data(1000)
    X_train, X_test, y_train, y_test = risk_system.prepare_data(df)
    risk_system.train_models(X_train, y_train)
    
    # Drift detector'ı başlat
    drift_detector = ModelDriftDetector()
    
    # Referans veri olarak eğitim setini kullan
    train_df = X_train.copy()
    train_df['high_risk'] = y_train
    drift_detector.set_reference_data(train_df)
    
    # Farklı drift seviyelerinde üretim verisi simüle et
    drift_levels = [0.0, 0.2, 0.5, 0.8]
    drift_history = []
    
    for i, drift_level in enumerate(drift_levels):
        print(f"\n--- Simülasyon {i+1}: Drift Seviyesi {drift_level} ---")
        
        # Üretim verisi oluştur
        production_data = drift_detector.generate_production_data(
            n_samples=200, 
            drift_factor=drift_level
        )
        
        # Veri drift tespiti
        print("Veri drift analizi yapılıyor...")
        data_drift_report = drift_detector.detect_data_drift(
            production_data, 
            save_report=False
        )
        
        # Drift testleri
        print("Drift testleri çalıştırılıyor...")
        test_results = drift_detector.run_drift_tests(production_data)
        
        # Model tahminleri
        X_prod = production_data[risk_system.feature_names]
        X_prod_scaled = risk_system.scaler.transform(X_prod)
        y_pred = risk_system.lr_model.predict(X_prod_scaled)
        
        # Performans izleme (gerçek etiketler varsa)
        if 'high_risk' in production_data.columns:
            performance = drift_detector.monitor_model_performance(
                production_data, y_pred
            )
        else:
            performance = None
        
        # Drift geçmişine ekle
        drift_entry = {
            'timestamp': datetime.now() - timedelta(days=30-i*7),
            'drift_score': drift_level,
            'feature_drifts': {
                'age': drift_level * 0.8,
                'bmi': drift_level * 0.6,
                'blood_pressure_systolic': drift_level * 0.9,
                'cholesterol': drift_level * 0.4
            }
        }
        
        if performance:
            drift_entry['model_performance'] = performance
        
        drift_history.append(drift_entry)
        
        # Uyarı sistemi
        if drift_level > 0.3:
            print("🚨 UYARI: Yüksek drift seviyesi tespit edildi!")
            print("   - Model yeniden eğitimi önerilir")
            print("   - Veri kalitesi kontrol edilmeli")
        else:
            print("✅ Drift seviyesi normal aralıkta")
    
    # Dashboard oluştur
    print("\nDrift izleme dashboard'u oluşturuluyor...")
    drift_detector.create_drift_dashboard(drift_history)
    
    return drift_detector, drift_history

def main():
    """Ana fonksiyon"""
    print("=== EVİDENTLY AI MODEL DRIFT TESPİTİ ===")
    
    # Drift izleme simülasyonunu çalıştır
    drift_detector, drift_history = simulate_drift_monitoring()
    
    print("\n=== ÖNERİLER ===")
    print("1. Düzenli olarak (günlük/haftalık) drift kontrolleri yapın")
    print("2. Drift tespit edildiğinde modeli yeniden eğitin")
    print("3. Veri kalitesi ve kaynak sistemleri kontrol edin")
    print("4. Otomatik uyarı sistemleri kurun")
    print("5. Model performansını sürekli izleyin")

if __name__ == "__main__":
    main()