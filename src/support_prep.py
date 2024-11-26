import pandas as pd
from tqdm import tqdm
import numpy as np

from itertools import product

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import OneHotEncoder

from category_encoders import TargetEncoder


from sklearn.ensemble import RandomForestRegressor

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from scipy.stats import chi2_contingency


def detectar_orden_cat(df, lista_cat, var_respuesta, sig_level = 0.05):
    for categoria in lista_cat:
        print(f"Estamos evaluando la variable {categoria.upper()}")
        df_crosstab = pd.crosstab(df[categoria], df[var_respuesta])
        display(df_crosstab)
        chi2, p, dof, expected = chi2_contingency(df_crosstab)

        if p<sig_level:
            print(f"Para la categoría {categoria.upper()} SÍ hay diferencias significativas, p = {p:.4f}")
            display(pd.DataFrame(expected, index = df_crosstab.index, columns = df_crosstab.columns).round())
        else:
            print(f"Para la categoría {categoria.upper()} NO hay diferencias significativas, p = {p:.4f}\n")
        print("--------"*10)

def normalize_scaler(data):
    """
    Normaliza los datos de un DataFrame utilizando una escala centrada en la media y ajustada al rango.
    
    Parameters:
        data (pd.DataFrame): DataFrame con datos numéricos a normalizar.
    
    Returns:
        pd.DataFrame: DataFrame con los datos normalizados.
    """
    data_copy = data.copy()
    for col in data.columns:
        mean_data = data_copy[col].mean()
        range_data = data_copy[col].max() - data_copy[col].min()
        data_copy[col] = data_copy[col].apply(lambda x: (x - mean_data) / range_data)
    return data_copy

def scale_data(data, columns, method="robust"):
    """
    Escala los datos de las columnas seleccionadas utilizando diferentes métodos de escalado.

    Parameters:
        data (pd.DataFrame): DataFrame con datos.
        columns (list): Lista de nombres de las columnas a escalar.
        method (str): Método de escalado ("minmax", "robust", "standard", "norm"). Default "robust".
    
    Returns:
        pd.DataFrame, object: DataFrame escalado y el scaler utilizado.
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "standard":
        scaler = StandardScaler()
    elif method == "norm":
        return normalize_scaler(data[columns]), None
    
    df_scaled = pd.DataFrame(scaler.fit_transform(data[columns]), columns=columns, index=data.index)
    return df_scaled, scaler

def find_outliers(data, columns, method = "ifo", random_state = 42, threshold = 70): 
    """
    Detecta outliers en un conjunto de datos utilizando el método especificado (Isolation Forest o Local Outlier Factor).
    
    Args:
        data (pd.DataFrame): El conjunto de datos sobre el cual se va a realizar la detección de outliers.
        columns (list): Lista de columnas en las cuales se desea detectar los outliers.
        method (str): Método para detectar outliers. Puede ser 'ifo' para Isolation Forest o 'lof' para Local Outlier Factor.
        random_state (int): Semilla para la aleatoriedad en el modelo de Isolation Forest.
        threshold (float): Porcentaje de outliers permitidos en los resultados. Si el porcentaje de outliers detectados en una fila es mayor a este valor, la fila se devuelve.

    Returns:
        pd.DataFrame: Un DataFrame con las filas que contienen una alta proporción de outliers.
        model: El modelo entrenado (Isolation Forest o Local Outlier Factor) que se puede reutilizar para nuevas predicciones.
    """
    df = data.copy()
    selected_data = df[columns]
    ests = np.linspace(1, 1000, 5, dtype=int)
    conts = np.linspace(0.01, 0.2, 5)
    neighs = np.linspace(15, 45, 5, dtype=int)
    
    if method == "ifo":   
        forest_arg_combis = list(product(ests, conts))
        for n, m in tqdm(forest_arg_combis):
            iforest = IsolationForest(random_state=random_state, n_estimators=n, contamination=m, n_jobs=-1)
            df[f"iforest_{n}_{m:.3f}"] = iforest.fit_predict(X=selected_data)
        df_detected = df.filter(like="iforest")
        model = iforest  # Devolvemos el modelo de Isolation Forest

    elif method == "lof":
        lof_combis = list(product(neighs, conts))
        for neighbour, contaminacion in tqdm(lof_combis):
            lof = LocalOutlierFactor(n_neighbors=neighbour, contamination=contaminacion, n_jobs=-1)
            df[f"lof_{neighbour}_{contaminacion:.3f}"] = lof.fit_predict(X=selected_data)
        df_detected = df.filter(like="lof_")
        model = lof  # Devolvemos el modelo de Local Outlier Factor

    percentages = df_detected.apply(percent_outs, axis=1)
    df_outliers = df[percentages > threshold] 

    # Eliminar las columnas creadas para detectar outliers
    df_outliers = df_outliers.drop(columns=df_outliers.filter(like="iforest").columns)
    df_outliers = df_outliers.drop(columns=df_outliers.filter(like="lof").columns)

    return df_outliers, model

def percent_outs(array):
    """
    Calcula el porcentaje de valores atípicos (-1) en un array.

    Parameters:
        array (np.array): Array con predicciones de detección de outliers.
    
    Returns:
        float: Porcentaje de valores atípicos.
    """
    length = len(array)
    neg_count = sum(array == -1)
    p_outs = neg_count / length * 100
    return p_outs