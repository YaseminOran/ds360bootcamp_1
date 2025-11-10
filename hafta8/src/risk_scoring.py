import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class HealthRiskScoring:
    def __init__(self):
        self.lr_model = LogisticRegression(random_state=42)
        self.rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def generate_sample_data(self, n_samples=1000):
        """Sentetik sağlık verisi oluşturur"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(50, 15, n_samples),
            'bmi': np.random.normal(25, 5, n_samples),
            'blood_pressure_systolic': np.random.normal(120, 20, n_samples),
            'blood_pressure_diastolic': np.random.normal(80, 10, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'glucose': np.random.normal(100, 20, n_samples),
            'smoking': np.random.binomial(1, 0.3, n_samples),
            'exercise_hours_week': np.random.exponential(3, n_samples),
            'family_history': np.random.binomial(1, 0.4, n_samples),
            'stress_level': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Yaş, BMI ve diğer faktörlere göre risk hesapla
        risk_score = (
            (df['age'] - 30) * 0.02 +
            (df['bmi'] - 20) * 0.1 +
            (df['blood_pressure_systolic'] - 120) * 0.01 +
            df['cholesterol'] * 0.002 +
            df['smoking'] * 0.5 +
            df['family_history'] * 0.3 +
            (5 - df['exercise_hours_week']) * 0.1 +
            df['stress_level'] * 0.1
        )
        
        # Risk skorunu 0-1 arasına normalize et
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
        
        # Yüksek risk etiketini oluştur (risk skoru > 0.6)
        df['high_risk'] = (risk_score > 0.6).astype(int)
        df['risk_score_continuous'] = risk_score
        
        return df
    
    def prepare_data(self, df):
        """Veriyi model eğitimi için hazırlar"""
        feature_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                       'cholesterol', 'glucose', 'smoking', 'exercise_hours_week',
                       'family_history', 'stress_level']
        
        X = df[feature_cols]
        y = df['high_risk']
        
        self.feature_names = feature_cols
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def train_models(self, X_train, y_train):
        """Logistic Regression ve Random Forest modellerini eğitir"""
        # Veriyi standartlaştır
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Modelleri eğit
        self.lr_model.fit(X_train_scaled, y_train)
        self.rf_model.fit(X_train, y_train)
        
        print("Modeller eğitildi!")
    
    def evaluate_models(self, X_test, y_test):
        """Modellerin performansını değerlendirir"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Tahminler
        lr_pred = self.lr_model.predict(X_test_scaled)
        rf_pred = self.rf_model.predict(X_test)
        
        lr_proba = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        rf_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        # ROC AUC skorları
        lr_auc = roc_auc_score(y_test, lr_proba)
        rf_auc = roc_auc_score(y_test, rf_proba)
        
        print("=== MODEL PERFORMANSI ===")
        print(f"\nLogistic Regression AUC: {lr_auc:.3f}")
        print(f"Random Forest AUC: {rf_auc:.3f}")
        
        print("\n=== LOGISTIC REGRESSION ===")
        print(classification_report(y_test, lr_pred))
        
        print("\n=== RANDOM FOREST ===")
        print(classification_report(y_test, rf_pred))
        
        return {
            'lr_auc': lr_auc,
            'rf_auc': rf_auc,
            'lr_pred': lr_pred,
            'rf_pred': rf_pred,
            'lr_proba': lr_proba,
            'rf_proba': rf_proba
        }
    
    def plot_feature_importance(self):
        """Random Forest özellik önemini görselleştirir"""
        if self.feature_names is None:
            print("Önce modeli eğitin!")
            return
            
        importance = self.rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.title('Random Forest - Özellik Önemi')
        plt.xlabel('Önem')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, y_test, results):
        """Karmaşıklık matrislerini görselleştirir"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Logistic Regression
        cm_lr = confusion_matrix(y_test, results['lr_pred'])
        sns.heatmap(cm_lr, annot=True, fmt='d', ax=axes[0], cmap='Blues')
        axes[0].set_title('Logistic Regression - Confusion Matrix')
        axes[0].set_ylabel('Gerçek')
        axes[0].set_xlabel('Tahmin')
        
        # Random Forest
        cm_rf = confusion_matrix(y_test, results['rf_pred'])
        sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[1], cmap='Blues')
        axes[1].set_title('Random Forest - Confusion Matrix')
        axes[1].set_ylabel('Gerçek')
        axes[1].set_xlabel('Tahmin')
        
        plt.tight_layout()
        plt.show()
    
    def predict_risk(self, patient_data):
        """Yeni hasta verisi için risk skorunu hesaplar"""
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
            
        patient_scaled = self.scaler.transform(patient_df[self.feature_names])
        
        lr_risk = self.lr_model.predict_proba(patient_scaled)[:, 1]
        rf_risk = self.rf_model.predict_proba(patient_df[self.feature_names])[:, 1]
        
        return {
            'logistic_regression_risk': lr_risk[0],
            'random_forest_risk': rf_risk[0],
            'average_risk': (lr_risk[0] + rf_risk[0]) / 2
        }

def main():
    """Ana fonksiyon - risk skorlama sistemini çalıştırır"""
    print("=== SAĞLIK RİSK SKORLAMA SİSTEMİ ===")
    
    # Risk skorlama sistemini başlat
    risk_system = HealthRiskScoring()
    
    # Sentetik veri oluştur
    print("\n1. Sentetik sağlık verisi oluşturuluyor...")
    df = risk_system.generate_sample_data(1000)
    print(f"Oluşturulan veri boyutu: {df.shape}")
    print(f"Yüksek risk oranı: %{df['high_risk'].mean()*100:.1f}")
    
    # Veriyi hazırla
    print("\n2. Veri eğitim ve test setlerine ayrılıyor...")
    X_train, X_test, y_train, y_test = risk_system.prepare_data(df)
    
    # Modelleri eğit
    print("\n3. Modeller eğitiliyor...")
    risk_system.train_models(X_train, y_train)
    
    # Modelleri değerlendir
    print("\n4. Model performansı değerlendiriliyor...")
    results = risk_system.evaluate_models(X_test, y_test)
    
    # Görselleştirmeler
    print("\n5. Görselleştirmeler oluşturuluyor...")
    risk_system.plot_feature_importance()
    risk_system.plot_confusion_matrices(y_test, results)
    
    # Örnek tahmin
    print("\n6. Örnek hasta risk tahmini...")
    sample_patient = {
        'age': 65,
        'bmi': 28,
        'blood_pressure_systolic': 140,
        'blood_pressure_diastolic': 90,
        'cholesterol': 240,
        'glucose': 120,
        'smoking': 1,
        'exercise_hours_week': 1,
        'family_history': 1,
        'stress_level': 4
    }
    
    risk_prediction = risk_system.predict_risk(sample_patient)
    print("\nÖrnek Hasta Profili:")
    for key, value in sample_patient.items():
        print(f"- {key}: {value}")
    
    print(f"\nRisk Skorları:")
    print(f"- Logistic Regression: {risk_prediction['logistic_regression_risk']:.3f}")
    print(f"- Random Forest: {risk_prediction['random_forest_risk']:.3f}")
    print(f"- Ortalama Risk: {risk_prediction['average_risk']:.3f}")
    
    if risk_prediction['average_risk'] > 0.6:
        print("⚠️  YÜKSEK RİSK - Doktor kontrolü önerilir")
    else:
        print("✅ DÜŞÜK RİSK - Normal takip")

if __name__ == "__main__":
    main()