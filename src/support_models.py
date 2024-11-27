import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,
    roc_auc_score, confusion_matrix
)
from IPython.display import display

class ClassificationModel:
    """
    Clase para crear, entrenar y evaluar modelos de clasificación.
    
    Attributes:
        X_train (array-like): Conjunto de características de entrenamiento.
        X_test (array-like): Conjunto de características de prueba.
        y_train (array-like): Etiquetas del conjunto de entrenamiento.
        y_test (array-like): Etiquetas del conjunto de prueba.
        model (estimator): Modelo de clasificación entrenado.
        metrics_df (DataFrame, optional): DataFrame con las métricas de evaluación.
        best_params (dict, optional): Los mejores parámetros encontrados durante la búsqueda de hiperparámetros.
    """
    def __init__(self, X, y, test_size=0.3, random_state=42):
        """
        Inicializa el modelo de clasificación y divide los datos en entrenamiento y prueba.

        Parameters:
            X (array-like): Características del conjunto de datos.
            y (array-like): Etiquetas del conjunto de datos.
            test_size (float, optional): Fracción de los datos a utilizar para el conjunto de prueba (por defecto 0.3).
            random_state (int, optional): Semilla para la aleatoriedad en la división de datos (por defecto 42).
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.model = None
        self.metrics_df = None
        self.best_params = None
        self.random_state = random_state
        self.resultados = {}
    
    def _get_model(self, model_type):
        """
        Obtiene el modelo seleccionado según el tipo indicado.

        Parameters:
            model_type (str): Tipo de modelo a usar ("logistic", "decision_tree", "random_forest", "gradient_boosting", "xgboost").

        Returns:
            estimator: Modelo de clasificación correspondiente al tipo seleccionado.

        Raises:
            ValueError: Si el tipo de modelo no es válido.
        """
        models = {
            "logistic": LogisticRegression(random_state=self.random_state),
            "decision_tree": DecisionTreeClassifier(random_state=self.random_state),
            "random_forest": RandomForestClassifier(random_state=self.random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
            "xgboost": XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        }
        if model_type not in models:
            raise ValueError(f"El modelo '{model_type}' no es válido. Elija uno de {list(models.keys())}")
        return models[model_type]

    def train(self, model_type, params=None, scoring='accuracy'):
        """
        Entrena el modelo seleccionado con los datos de entrenamiento y calcula las métricas de evaluación.

        Parameters:
            model_type (str): Tipo de modelo a usar ("logistic", "decision_tree", "random_forest", "gradient_boosting", "xgboost").
            params (dict, optional): Parámetros para la búsqueda en cuadrícula (por defecto None).

        Returns:
            estimator: Modelo de clasificación entrenado.
        """
        self.model = self._get_model(model_type)
        
        if params:
            grid_search = GridSearchCV(self.model, param_grid=params, cv=5, scoring=scoring, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
        else:
            self.model.fit(self.X_train, self.y_train)
        
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Guardar predicciones, modelo y métricas
        self.resultados[model_type] = {
            "pred_train": y_train_pred,
            "pred_test": y_test_pred,
            "mejor_modelo": self.model,
            "metrics": self.calcular_metricas(y_train_pred, y_test_pred)
        }

        self.metrics_df = self.resultados[model_type]['metrics']
        
        return self.model

    def calcular_metricas(self, y_train_pred, y_test_pred):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC, Kappa,
        tiempo de computación y núcleos utilizados.
        
        Parameters:
            y_train_pred (array-like): Predicciones del conjunto de entrenamiento.
            y_test_pred (array-like): Predicciones del conjunto de prueba.
        
        Returns:
            DataFrame: DataFrame con las métricas para los conjuntos de entrenamiento y prueba.
        """
        modelo = self.model

        # Registrar tiempo de ejecución
        start_time = time.time()
        if hasattr(modelo, "predict_proba"):
            prob_train = modelo.predict_proba(self.X_train)[:, 1]
            prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = prob_test = None
        elapsed_time = time.time() - start_time

        # Cálculo de métricas
        metrics = {
            'precision' : [precision_score(self.y_train, y_train_pred), precision_score(self.y_test, y_test_pred)],
            'accuracy' : [accuracy_score(self.y_train, y_train_pred), accuracy_score(self.y_test, y_test_pred)],
            'recall' : [recall_score(self.y_train, y_train_pred), recall_score(self.y_test, y_test_pred)],
            'f1_score' : [f1_score(self.y_train, y_train_pred), f1_score(self.y_test, y_test_pred)],
            'kappa': [cohen_kappa_score(self.y_train, y_train_pred), cohen_kappa_score(self.y_test, y_test_pred)],
            'auc': [roc_auc_score(self.y_train, prob_train) if prob_train is not None else None, roc_auc_score(self.y_test, prob_test) if prob_test is not None else None],
            'time' : elapsed_time,
            'n_jobs': [getattr(modelo, "n_jobs", psutil.cpu_count(logical=True))] * 2
        }
        df_metrics = pd.DataFrame(metrics, columns=metrics.keys(), index=['train', 'test'])
        return df_metrics

    def display_metrics(self):
        """
        Muestra las métricas de evaluación del modelo.

        Si las métricas no están disponibles, muestra un mensaje indicándolo.
        """
        if self.metrics_df is not None:
            display(self.metrics_df)
        else:
            print("No hay métricas disponibles. Primero entrena el modelo.")
    
    def plot_confusion_matrix(self):
        """
        Muestra la matriz de confusión para los conjuntos de entrenamiento y prueba.

        Si el modelo no ha sido entrenado previamente, muestra un mensaje indicándolo.
        """
        if self.model is None:
            print("Primero debes entrenar un modelo para graficar la matriz de confusión.")
            return
        
        y_test_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_test_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    
    def get_best_params(self):
        """
        Obtiene los mejores parámetros del modelo si se ha realizado una búsqueda en cuadrícula.

        Returns:
            dict or None: Diccionario con los mejores parámetros si se realizó la búsqueda en cuadrícula, 
                        o `None` si no hay parámetros disponibles.
        """
        if self.best_params:
            return self.best_params
        else:
            print("No se ha realizado búsqueda en cuadrícula o no hay parámetros disponibles.")
            return None

    def return_model(self):
        """
        Retorna el modelo actual.

        Returns:
            estimator: El modelo entrenado o el modelo base usado en la instancia.
        """
        return self.model
