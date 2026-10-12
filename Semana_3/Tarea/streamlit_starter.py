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


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption("Explora qué porcentaje de clientes conviene contactar según el uplift estimado y el valor esperado de la campaña.")

df = load_data()

# TODO: agreguen un slider en la barra lateral para el % de la base a contactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider("Porcentaje de clientes a contactar", 0, 100, 50)

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- Curva Qini del T-learner ---
def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    tratados = (orden["treatment_group"] == "treatment").values
    respuesta = orden["responded_60d"].values

    respuestas_t = np.cumsum(np.where(tratados, respuesta, 0))
    respuestas_c = np.cumsum(np.where(~tratados, respuesta, 0))
    total_t = np.cumsum(tratados)
    total_c = np.cumsum(~tratados)

    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = total_t[i - 1], total_c[i - 1]
        tasa_t = respuestas_t[i - 1] / nt if nt > 0 else 0
        tasa_c = respuestas_c[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


xs, ys = curva_qini(test, "uplift_estimado")
qini = coeficiente_qini(xs, ys, len(test))

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(xs, ys, color=GREEN, linewidth=3, label=f"T-learner (Qini={qini:+.3f})")
ax.plot([0, 1], [0, ys[-1]], linestyle="--", color=GRAY, label="Selección al azar")
ax.set_xlabel("Proporción de clientes contactados")
ax.set_ylabel("Respuestas incrementales acumuladas")
ax.set_title("Curva Qini del T-learner")
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --- Tabla de políticas para el porcentaje elegido ---
n_contactados = int(len(test) * pct_contactar / 100)
seleccionados = test.sort_values("uplift_estimado", ascending=False).head(n_contactados).index
valor_cliente, cantidad = valor_de_la_politica(test, seleccionados)

tabla_politicas = pd.DataFrame([{
    "pct_contactado": pct_contactar,
    "valor_por_cliente": round(valor_cliente, 2),
    "n_contactados": cantidad,
    "valor_total": round(valor_cliente * cantidad, 2),
}])

st.subheader(f"Tabla de políticas para {pct_contactar}% de la base contactada")
st.dataframe(tabla_politicas)