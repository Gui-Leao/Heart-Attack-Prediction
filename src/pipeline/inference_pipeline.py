import os
import logging
import pandas as pd
from pydantic import BaseModel
from pipeline.model_pipeline import HeartAttackModel
from pipeline.feature_eng_pipeline import HeartAttackPreprocessor



logger = logging.getLogger(__name__)

class PredictPipeline:
    """
    Classe para utilizar o modelo e o preprocessamento dos dados.
    """

    def __init__(self, model_path = None,pipeline_path = None):
        project_root = model_path or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = pipeline_path or os.path.join(project_root,"artifacts/heart_attack_model.pkl")
        self.preprocessor_path = os.path.join(project_root,"artifacts/preprocessor.pkl")
        self.model = None
        self.preprocessor = None
        self._load_artifacts()


    def _load_artifacts(self):
        logger.info("Carregando modelo e preprocessor...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor não encontrado em: {self.preprocessor_path}")
        
        self.model = HeartAttackModel.load(self.model_path)
        self.preprocessor = HeartAttackPreprocessor.load(self.preprocessor_path)
        logger.info("Modelo e preprocessor carregados com sucesso.")

    def predict(self, features: pd.DataFrame):
        try:
            logger.info('Pré-processando dados de entrada...')
            processed = self.preprocessor.transform(features)
            logger.info('Realizando predição...')
            pred = self.model.predict(processed)[0]
            proba = self.model.predict_proba(processed)[0][1]
            return int(pred), float(proba)
        except Exception as e:
            logger.error(f'Erro na predição: {e}')
            raise
    
class InputData:
    """
    Classe para estruturar os dados de entrada para predição.
    """

    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)  

    def to_dataframe(self):
        """Converte os dados de entrada em um DataFrame do pandas"""
        return pd.DataFrame([self.__dict__])
    


if __name__ == "__main__":
    
    
    input_data = InputData(age=10,sex=1,cp=2,trestbps=120,chol=240,fbs=0,restecg=1,thalach=150,exang=0,oldpeak=1.5,slope=2,ca=0,thal=3)
    df = input_data.to_dataframe()
    pipeline = PredictPipeline()
    pipeline._load_artifacts()
    prediction, probability = pipeline.predict(df)
    print(f"Predição: {prediction}, Probabilidade: {probability}")
    logger.info(f"Predição: {prediction}, Probabilidade: {probability}")
    