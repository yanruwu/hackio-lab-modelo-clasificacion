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
    
import numpy as np
import pandas as pd
import plotly.express as px
def plot_interactive_probs(y_real,y_model_prob):
    # Crear dataframe
    data = pd.DataFrame({
        "Predicted Probability": y_model_prob,
        "True Label": y_real,
        "True Value": y_real.astype(str),
        "Jitter": np.random.uniform(-0.4, 0.4, len(y_model_prob))
    })

    # Función para calcular métricas con un threshold dado
    def compute_metrics(threshold):
        tn = ((data["Predicted Probability"] <= threshold) & (data["True Label"] == 0)).sum()
        tp = ((data["Predicted Probability"] > threshold) & (data["True Label"] == 1)).sum()
        all_p = (data["Predicted Probability"] > threshold).sum()
        all_n = (data["Predicted Probability"] <= threshold).sum()
        recall = tp / all_p if all_p > 0 else 0
        specificity = tn / all_n if all_n > 0 else 0
        return recall, specificity

    # Inicializar el threshold
    initial_threshold = 0.5
    recall, specificity = compute_metrics(initial_threshold)

    # Crear scatterplot con Plotly Express
    fig = px.scatter(
        data, 
        x="Predicted Probability", 
        y="Jitter", 
        color="True Value",
        color_discrete_map={"0": "red", "1": "green"},  # Colores personalizados
        opacity=0.6,
        title="Threshold and Metrics"
    )



    # Añadir áreas sombreadas (zonas roja y verde)
    fig.add_shape(
        type="rect",
        x0=0, x1=initial_threshold,
        y0=-1, y1=1,
        fillcolor="red",
        opacity=0.3,
        layer="below",
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=initial_threshold, x1=1,
        y0=-1, y1=1,
        fillcolor="green",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    # Añadir línea vertical inicial
    fig.add_vline(
        x=initial_threshold, 
        line_width=2, 
        line_dash="dash", 
        line_color="red",
        annotation_text="",
        annotation_position="top left"
    )

    # Añadir anotaciones iniciales
    fig.add_annotation(
        x=1, y=1.1,
        xref="paper",  # Usar coordenadas relativas al "paper" (el área fuera del gráfico)
        yref="paper",  # Coordenada relativa al "paper" 
        text=f"Recall (TPR): {recall:.3f}", 
        showarrow=False, 
        font=dict(size=14, family = 'Arial Black')
    )
    fig.add_annotation(
        x=0, y=1.1, 
        xref="paper",  # Usar coordenadas relativas al "paper" (el área fuera del gráfico)
        yref="paper",  # Coordenada relativa al "paper" 
        text=f"Specificity (TNR): {specificity:.3f}", 
        showarrow=False, 
        font=dict(size=14, family = 'Arial Black')
    )

    # Configurar slider para actualizar threshold
    threshold_values = [n for n in np.arange(min(y_model_prob)*0.99, max(y_model_prob)*1.01+0.001,0.0002)]

    sliders = [dict(
        steps=[
            dict(
                method="relayout",
                args=[
                    {
                        # Actualizar línea vertical
                        "shapes[2].x0": t,
                        "shapes[2].x1": t,
                        # Actualizar área roja
                        "shapes[0].x1": t,
                        # Actualizar área verde
                        "shapes[1].x0": t,
                        # Actualizar anotaciones
                        "annotations[1].text": f"Recall (TPR): {compute_metrics(t)[0]:.3f}",
                        "annotations[2].text": f"Specificity (TNR): {compute_metrics(t)[1]:.3f}"
                    }
                ],
                label=f"{t:.3f}"
            )
            for t in threshold_values
        ],
        active=0.5,
        currentvalue={"prefix": "Threshold: "},
        pad={"t": 50},
    )]

    # Añadir slider a la figura
    fig.update_layout(sliders=sliders)
    fig.update(layout_coloraxis_showscale=False)

    fig.update_layout(
        xaxis=dict(
            range=[min(y_model_prob)*0.99, max(y_model_prob)*1.01] # Cambiar el rango del eje X
        ),
        yaxis=dict(
            range=[-0.8, 0.8],  # Cambiar el rango del eje X
            showticklabels=False
        ),
        yaxis_title=None
    )

    # Mostrar gráfica interactiva
    fig.show()
