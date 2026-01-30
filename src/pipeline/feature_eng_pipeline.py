import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer():

    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Feature: age * trestbps (pressão arterial em idosos é mais preocupante)
        X_copy['age_trestbps'] = X_copy['age'] * X_copy['trestbps']
        
        # Feature: chol / age (colesterol relativo à idade)
        X_copy['chol_age_ratio'] = X_copy['chol'] / (X_copy['age'] + 1)
        
        # Feature: thalach - age (frequência cardíaca relativa à idade)
        X_copy['thalach_age_diff'] = X_copy['thalach'] - X_copy['age']
        
        return X_copy
    

class HeartAttackPreprocessor:
    """
    Pipeline completo de preprocessing para dados de ataque cardíaco
    """
    
    def __init__(self):
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        self.numerical_features = None
        self.pipeline = None
        self.feature_names_in_ = None
        
    def _build_pipeline(self, X: pd.DataFrame):
        """Constrói o pipeline de transformação"""
        
        # Identificar features numéricas (todas exceto as categóricas)
        all_features = X.columns.tolist()
        numerical_features = [f for f in all_features if f not in self.categorical_features]
        # Evitar duplicidade de colunas
        self.numerical_features = numerical_features
        
        # Ordinal encoding para 'slope' (preserva ordem)
        ordinal_features = ['slope']
        
        # One-hot encoding para outras categóricas
        onehot_features = [f for f in self.categorical_features if f != 'slope']
        
        # Transformadores
        feature_engineer = FeatureEngineer()
        
        numerical_transformer = Pipeline(steps=[
            ('feature_engineer', feature_engineer),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = ColumnTransformer(
            transformers=[
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), onehot_features)
            ]
        )
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
    def fit(self, X: pd.DataFrame, y=None):
        """Treina o preprocessor"""
        self.feature_names_in_ = X.columns.tolist()
        self._build_pipeline(X)
        self.pipeline.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame):
        """Transforma os dados"""
        if self.pipeline is None:
            raise ValueError("Preprocessor não foi treinado. Use fit() primeiro.")
        
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        """Treina e transforma"""
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath: str):
        """Salva o preprocessor"""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str):
        """
        Carrega um modelo salvo
        
        Args:
            filepath: Caminho do preprocessor salvo
        
        Returns:
            Preprocessor carregado
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor carregado de: {filepath}")
        return preprocessor


def create_heart_attack_preprocessor(X_train: pd.DataFrame) -> HeartAttackPreprocessor:
    """
    Factory function para criar e treinar o preprocessor
    
    Args:
        X_train: DataFrame de treinamento
    
    Returns:
        Preprocessor treinado
    """
    preprocessor = HeartAttackPreprocessor()
    preprocessor.fit(X_train)
    return preprocessor