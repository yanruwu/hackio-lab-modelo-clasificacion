from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
import pandas as pd


def metricas_logisticas(y_train, y_train_pred, y_test, y_test_pred, train_prob = None, test_prob = None):
    metrics = {
    'precision' : [precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)],
    'accuracy' : [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)],
    'recall' : [recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)],
    'f1_score' : [f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)],
    'kappa': [cohen_kappa_score(y_train, y_train_pred), cohen_kappa_score(y_test, y_test_pred)],
    'auc': [roc_auc_score(y_train, train_prob) if train_prob is not None else None, roc_auc_score(y_test, test_prob) if test_prob is not None else None]
    }
    df_metrics = pd.DataFrame(metrics, columns=metrics.keys(), index = ["train", "test"])
    return df_metrics
    