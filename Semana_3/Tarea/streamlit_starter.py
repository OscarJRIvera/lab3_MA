"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "data" / "uplift_campaign.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_path}")
    return pd.read_csv(data_path)


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)

def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)

    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_tratados_acum[i - 1], n_control_acum[i - 1]
        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n

# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption("Esta app permite explorar el incremento en ventas de un programa de fidelización de Starbucks, usando un modelo de T-learner, buscando el % de la base a contactar que maximice el valor de la política.")


df = load_data()

def tabla_de_politicas(test, pct_contactar):
    resultados = []
    n_c = int(len(test) * pct_contactar / 100)
    idx = test.sort_values("uplift_estimado", ascending=False).head(n_c).index
    valor_cliente, n = valor_de_la_politica(test, idx)
    resultados.append({"pct_contactado": pct_contactar, "valor_por_cliente": round(valor_cliente, 2), "n_contactados": n, "valor_total": round(valor_cliente * n, 2)})

    tabla_politicas = pd.DataFrame(resultados)
    st.subheader(f"Tabla de políticas para {pct_contactar}% de la base contactada")
    st.dataframe(tabla_politicas)

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)
# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider("Porcentaje de la base a contactar", 0, 100, 50)   # <-- reemplacen esto: st.slider(...)
    tabla_de_politicas(test, pct_contactar)


# --- TODO: calculen y grafiquen la curva Qini de su T-learner ---

xs_uplift, ys_uplift = curva_qini(test, "uplift_estimado")
qini_uplift = coeficiente_qini(xs_uplift, ys_uplift, len(test))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xs_uplift, ys_uplift, color="#00704A", linewidth=3, label=f"T-learner (Qini={qini_uplift:+.3f})")
ax.plot([0, 1], [0, ys_uplift[-1]], color="#9A988E", linestyle="--", label="Al azar")
ax.set_xlabel("% de la base contactada (de prueba)")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.set_title("Curva Qini de T-learner")
ax.legend()
plt.tight_layout()
plt.savefig("curva_qini_streamlit.png", dpi=150)
plt.show()

img_path = "curva_qini_streamlit.png"
st.image(img_path)

# --- TODO: muestren la tabla de políticas (valor por cliente y total) para el % elegido ---


