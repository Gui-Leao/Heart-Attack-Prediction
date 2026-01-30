"""
Script de treinamento e deployment do modelo
"""


import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
import sys

# Adiciona src ao path
# sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.feature_eng_pipeline  import HeartAttackPreprocessor, create_heart_attack_preprocessor
from pipeline.model_pipeline import HeartAttackModel


logger = logging.getLogger(__name__)


def train_and_save_model(data_path: str, output_dir: str = "artifacts"):
    """
    Treina o modelo completo e salva os artefatos
    
    Args:
        data_path: Caminho para o arquivo CSV de dados
        output_dir: Diretório para salvar o modelo
        model_type: Tipo de modelo a treinar
    """
    
    logger.info("=" * 60)
    logger.info("INICIANDO TREINAMENTO E DEPLOYMENT DO MODELO")
    logger.info("=" * 60)
    
    # Criar diretório de output
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Carregar dados
    logger.info(f"Carregando dados de: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Separar features e target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Split treino/teste
    logger.info("Dividindo dados em treino (80%) e teste (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")
    
    # 3. Criar e treinar preprocessor
    logger.info("Criando e treinando preprocessor...")
    preprocessor = create_heart_attack_preprocessor(X_train)
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    logger.info(f"Preprocessor salvo em: {preprocessor_path}")
    
    # 4. Preprocessar dados
    logger.info("Preprocessando dados...")
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info(f"Dados processados: {X_train_processed.shape}")
    
    # 5. Treinar modelo
    logger.info(f"Treinando modelo de Regressão Logística com hiperparâmetros ótimos...")
    model = HeartAttackModel()
    cv_metrics = model.train(pd.DataFrame(X_train_processed), y_train,preprocessor.feature_names_in_)
    logger.info(f"CV F1 Score: {cv_metrics['cv_f1_mean']:.4f} (+/- {cv_metrics['cv_f1_std']:.4f})")
    
    # 6. Avaliar modelo
    logger.info("Avaliando modelo...")
    eval_metrics = model.evaluate(pd.DataFrame(X_test_processed), y_test)
    logger.info(f"F1 Score (teste): {eval_metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC Score: {eval_metrics['roc_auc_score']:.4f}")
    
    # Print relatório de classificação
    logger.info("\nRelatório de Classificação:")
    logger.info(eval_metrics['classification_report'])
    
    # 7. Salvar modelo
    model_path = os.path.join(output_dir, "heart_attack_model.pkl")
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    model.save(model_path, metadata_path)
    logger.info(f"Modelo salvo em: {model_path}")
    
    logger.info("=" * 60)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 60)
    
    return model, preprocessor, eval_metrics


def main():
    """Script principal"""
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "dataset", "Heart_Attack_Data_Set.csv")
    output_dir = os.path.join(project_root, "artifacts")

    if not os.path.exists(data_path):
        logger.error(f"Arquivo de dados não encontrado: {data_path}")
        return

    # Treinar e salvar modelo
    try:
        train_and_save_model(
            data_path=data_path,
            output_dir=output_dir
        )
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}", exc_info=True)


if __name__ == "__main__":
    main()
