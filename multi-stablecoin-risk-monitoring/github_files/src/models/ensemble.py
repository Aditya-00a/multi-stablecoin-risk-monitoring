"""
Weighted Ensemble Model for Stablecoin Risk Detection

Combines three models:
- Isolation Forest (35%): Unsupervised anomaly detection
- One-Class SVM (25%): Boundary-based anomaly detection  
- XGBoost (40%): Supervised gradient boosting

Author: Aditya Sakhale
Institution: NYU School of Professional Studies
Date: November 2025
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Optional


class EnsembleModel:
    """
    Weighted ensemble combining Isolation Forest, One-Class SVM, and XGBoost
    for stablecoin transaction risk detection.
    """
    
    def __init__(
        self,
        weight_if: float = 0.35,
        weight_svm: float = 0.25,
        weight_xgb: float = 0.40
    ):
        """
        Initialize ensemble with specified weights.
        
        Args:
            weight_if: Isolation Forest weight (default 0.35)
            weight_svm: One-Class SVM weight (default 0.25)
            weight_xgb: XGBoost weight (default 0.40)
        """
        # Validate weights sum to 1
        total = weight_if + weight_svm + weight_xgb
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.weights = {
            "isolation_forest": weight_if,
            "svm": weight_svm,
            "xgboost": weight_xgb
        }
        
        # Initialize models
        self.isolation_forest = None
        self.svm = None
        self.xgboost = None
        self.scaler = StandardScaler()
        
        self._models_loaded = False
    
    def is_loaded(self) -> bool:
        """Check if models are loaded"""
        return self._models_loaded
    
    def initialize_models(self):
        """Initialize models with default parameters"""
        
        # Isolation Forest
        # - n_estimators=100: Number of trees
        # - contamination='auto': Automatically determine anomaly threshold
        # - max_samples=256: Samples per tree for efficiency
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination='auto',
            max_samples=256,
            random_state=42,
            n_jobs=-1
        )
        
        # One-Class SVM
        # - kernel='rbf': Radial basis function for non-linear boundaries
        # - gamma='scale': Automatic gamma calculation
        # - nu=0.1: Upper bound on fraction of outliers
        self.svm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        
        # XGBoost Classifier
        # - scale_pos_weight=99: Handle class imbalance (<1% positive cases)
        # - max_depth=5: Prevent overfitting
        # - learning_rate=0.1: Standard learning rate
        self.xgboost = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=99,  # Critical for class imbalance
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        )
        
        self._models_loaded = True
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit all models in the ensemble.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels for XGBoost (optional for IF and SVM)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit unsupervised models
        print("Fitting Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        print("Fitting One-Class SVM...")
        self.svm.fit(X_scaled)
        
        # Fit XGBoost (requires labels)
        if y is not None:
            print("Fitting XGBoost...")
            self.xgboost.fit(X_scaled, y)
        else:
            print("Warning: No labels provided, XGBoost not fitted")
        
        self._models_loaded = True
        print("Ensemble training complete!")
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Generate ensemble risk prediction.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Dictionary with risk_score, confidence, and model contributions
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert features to array
        feature_names = [
            'mint_burn_ratio', 'concentration_index', 'realized_volatility',
            'net_exchange_flow', 'tx_value_ratio', 'cross_asset_corr',
            'whale_activity', 'volume_zscore', 'supply_change_rate',
            'holder_growth_rate', 'hour_sin', 'hour_cos', 'day_of_week',
            'treasury_yield_2y', 'treasury_spread'
        ]
        
        X = np.array([[features.get(f, 0.0) for f in feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # Get individual model scores
        # Isolation Forest: Convert decision function to 0-1 probability
        if_score = self.isolation_forest.decision_function(X_scaled)[0]
        if_prob = 1 / (1 + np.exp(if_score))  # Sigmoid transformation
        
        # One-Class SVM: Convert decision function to 0-1 probability
        svm_score = self.svm.decision_function(X_scaled)[0]
        svm_prob = 1 / (1 + np.exp(svm_score))  # Sigmoid transformation
        
        # XGBoost: Direct probability
        xgb_prob = self.xgboost.predict_proba(X_scaled)[0][1]
        
        # Weighted ensemble
        ensemble_score = (
            self.weights["isolation_forest"] * if_prob +
            self.weights["svm"] * svm_prob +
            self.weights["xgboost"] * xgb_prob
        )
        
        # Calculate confidence (inverse of prediction variance)
        scores = [if_prob, svm_prob, xgb_prob]
        variance = np.var(scores)
        confidence = 1 - min(variance * 4, 1.0)  # Scale variance to confidence
        
        return {
            "risk_score": float(ensemble_score),
            "confidence": float(confidence),
            "model_contributions": {
                "isolation_forest": float(if_prob),
                "svm": float(svm_prob),
                "xgboost": float(xgb_prob)
            }
        }
    
    def save_models(self, path: str = "./models/saved/"):
        """Save all models to disk"""
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.isolation_forest, os.path.join(path, "isolation_forest.joblib"))
        joblib.dump(self.svm, os.path.join(path, "svm.joblib"))
        joblib.dump(self.xgboost, os.path.join(path, "xgboost.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))
        
        print(f"Models saved to {path}")
    
    def load_models(self, path: str = "./models/saved/"):
        """Load models from disk or initialize new ones"""
        try:
            self.isolation_forest = joblib.load(os.path.join(path, "isolation_forest.joblib"))
            self.svm = joblib.load(os.path.join(path, "svm.joblib"))
            self.xgboost = joblib.load(os.path.join(path, "xgboost.joblib"))
            self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
            self._models_loaded = True
            print(f"Models loaded from {path}")
        except FileNotFoundError:
            print("No saved models found, initializing new models...")
            self.initialize_models()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get XGBoost feature importance scores"""
        if self.xgboost is None:
            return {}
        
        feature_names = [
            'mint_burn_ratio', 'concentration_index', 'realized_volatility',
            'net_exchange_flow', 'tx_value_ratio', 'cross_asset_corr',
            'whale_activity', 'volume_zscore', 'supply_change_rate',
            'holder_growth_rate', 'hour_sin', 'hour_cos', 'day_of_week',
            'treasury_yield_2y', 'treasury_spread'
        ]
        
        importance = self.xgboost.feature_importances_
        return dict(zip(feature_names, importance))


# Example usage
if __name__ == "__main__":
    # Initialize ensemble
    ensemble = EnsembleModel(
        weight_if=0.35,
        weight_svm=0.25,
        weight_xgb=0.40
    )
    
    # Initialize models
    ensemble.initialize_models()
    
    # Example prediction
    sample_features = {
        'mint_burn_ratio': 1.2,
        'concentration_index': 0.45,
        'realized_volatility': 0.03,
        'net_exchange_flow': -50000,
        'tx_value_ratio': 2.5,
        'cross_asset_corr': 0.72,
        'whale_activity': 1,
        'volume_zscore': 1.8,
        'supply_change_rate': 0.02,
        'holder_growth_rate': 0.01,
        'hour_sin': 0.5,
        'hour_cos': 0.866,
        'day_of_week': 3,
        'treasury_yield_2y': 4.5,
        'treasury_spread': 0.8
    }
    
    print("Sample prediction:")
    print(ensemble.predict(sample_features))
