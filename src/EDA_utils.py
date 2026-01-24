import pandas as pd
import scipy.stats as stats

def showOutliers(df:pd.DataFrame,numerical_features:list)->None:
    '''
        Função para mostrar o percentual de outliers de cada variável numérica
        Args:
            df: DataFrame com os dados
            numerical_features: Lista de variáveis numéricas
        Returns:
            None
    '''
    for num, feature in enumerate(numerical_features):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1


        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR


        outliers = df[(df[feature] < limite_inferior) | (df[feature] > limite_superior)]

        percentual_outliers = (len(outliers) / len(df)) * 100
        print(f'Percentual de outliers {feature} (IQR): {percentual_outliers:.2f}%')


def checkNormality(data:list)->None:
    '''
    Realiza o Teste de Shapiro-Wilk para normalidade.
    Args:
        data: Lista de dados
    Returns:
        None
    '''
    
    test_stat, p_value = stats.shapiro(data)  # Teste de normalidade
    print(f"p-valor: {p_value:.4f}")
    
    if p_value < 0.05:
        print("🔴 Rejeita H₀: Os dados NÃO são normalmente distribuídos.")
    else:
        print("🟢 Falhou em rejeitar H₀: Os dados PODEM ser normalmente distribuídos.")



def checkMannWhitney(group1:list, group2:list)->None:
    '''
    Realiza o Teste de Mann-Whitney U para comparar se as distribuições de dois grupos são diferentes.
    Args:
        group1: Lista de dados do primeiro grupo
        group2: Lista de dados do segundo grupo
    Returns:
        None
    '''
    test_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')  # Teste bilateral

    print(f"p-valor: {p_value:.4f}")

    if p_value < 0.05:
        print("🔴 Rejeita H₀: As distribuições dos grupos são diferentes.")
    else:
        print("🟢 Falhou em rejeitar H₀: Não há evidência de que as distribuições sejam diferentes.")

