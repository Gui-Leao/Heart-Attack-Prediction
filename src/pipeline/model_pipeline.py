"""
Módulo de modelo para Heart Attack Prediction
Encapsula o treinamento, avaliação e inferência
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import joblib
from typing import Dict, Any, Tuple
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HeartAttackModel:
    """
    Wrapper para o modelo de previsão de ataque cardíaco
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa o modelo de Regressão Logística com hiperparâmetros ótimos.
        Args:
            random_state: Seed para reprodutibilidade
        """
        self.random_state = random_state
        self.model_type = 'LogisticRegression'
        self.model = LogisticRegression(
            C=0.4315947185115295,
            penalty='l2',
            random_state=self.random_state,
            max_iter=1000
        )
        self.is_trained = False
        self.metrics = {}
        self.feature_names_ = None
        self.training_date = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series , feature_names: list[str]) -> Dict[str, Any]:
        """
        Treina o modelo
        
        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
        
        Returns:
            Dicionário com métricas de cross-validation
        """
        self.feature_names_ = feature_names
        
        logger.info(f"Treinando modelo {self.model_type}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.training_date = datetime.now().isoformat()
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='f1'
        )
        
        self.metrics['cv_f1_scores'] = cv_scores.tolist()
        self.metrics['cv_f1_mean'] = float(cv_scores.mean())
        self.metrics['cv_f1_std'] = float(cv_scores.std())
        
        logger.info(f"CV F1 Score: {self.metrics['cv_f1_mean']:.4f} (+/- {self.metrics['cv_f1_std']:.4f})")
        
        return self.metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
        
        Returns:
            Dicionário com métricas de avaliação
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Use train() primeiro.")
        
        logger.info("Avaliando modelo...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc_score': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.metrics.update(metrics)
        
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC Score: {metrics['roc_auc_score']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Features para predição
        
        Returns:
            Array com as predições (0 ou 1)
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Use train() primeiro.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições com probabilidades
        
        Args:
            X: Features para predição
        
        Returns:
            Array com as probabilidades para cada classe
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Use train() primeiro.")
        
        return self.model.predict_proba(X)
    
    def save(self, model_path: str, metadata_path: str = None):
        """
        Salva o modelo e metadados
        
        Args:
            model_path: Caminho para salvar o modelo
            metadata_path: Caminho para salvar metadados (opcional)
        """
        joblib.dump(self.model, model_path)
        logger.info(f"Modelo salvo em: {model_path}")
        
        if metadata_path:
            metadata = {
                'model_type': self.model_type,
                'is_trained': self.is_trained,
                'training_date': self.training_date,
                'feature_names': self.feature_names_,
                'metrics': self.metrics
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadados salvos em: {metadata_path}")
    
    @staticmethod
    def load(model_path: str):
        """
        Carrega um modelo salvo
        
        Args:
            model_path: Caminho do modelo salvo
        
        Returns:
            Modelo carregado
        """
        model = joblib.load(model_path)
        logger.info(f"Modelo carregado de: {model_path}")
        return model
