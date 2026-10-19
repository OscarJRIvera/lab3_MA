"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"


@st.cache_data
def cargar():
    catalogo = pd.read_csv("data/catalogo.csv")
    hist = pd.read_csv("data/ordenes_historial.csv")
    hold = pd.read_csv("data/ordenes_holdout.csv")
    exp = pd.read_csv("data/experimento.csv")
    clientes = pd.read_csv("data/clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)


def puntajes(metodo, x_vec, K, cliente_idx=None):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it
    # TODO 3: filtrado por cliente: similitud del cliente con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    sims = cosine_similarity(x_vec[None, :], X)[0]
    if cliente_idx is not None:
        sims[cliente_idx] = 0  # un cliente no puede ser su propio vecino
    vecinos = np.argsort(sims)[-K:]
    return sims[vecinos] @ X[vecinos]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)   # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf                                     # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption("Explora qué productos recomendar a cada cliente y compara cuál método acierta más y ofrece mayor variedad.")

# TODO 4: barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox("Método", ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"])
    K = st.slider("Número de vecinos (K)", 5, 100, 30, 5)

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.selectbox("ID de cliente", options=[int(i) for i in ids], index=int(np.where(ids == 14)[0][0]) if 14 in ids else 0)
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    st.write("Ya compró:", [productos[j] for j in np.where(x_vec > 0)[0]])

    if not x_vec.any() and metodo != "Popularidad":
        st.info("Este cliente no tiene historial de compras. Se usa popularidad como respaldo.")
        metodo_efectivo = "Popularidad"
    else:
        metodo_efectivo = metodo

    scores = puntajes(metodo_efectivo, x_vec, K, fila)
    recomendaciones = top3(scores, x_vec)

    if metodo_efectivo == "Filtrado por cliente":
        similitudes = cosine_similarity(x_vec[None, :], X)[0]
        similitudes[fila] = 0
        vecinos = np.argsort(similitudes)[-K:]

    st.subheader("Tus 3 recomendaciones")
    for posicion, j in enumerate(recomendaciones, 1):
        if metodo_efectivo == "Popularidad":
            razon = "Es uno de los productos más comprados por los clientes."
        elif metodo_efectivo == "Filtrado por ítem":
            comprados = np.where(x_vec > 0)[0]
            relacionado = comprados[np.argmax(S_it[comprados, j])]
            razon = f"Porque compró {productos[relacionado]} (similitud {S_it[relacionado, j]:.2f})."
        else:
            cuantos = int(X[vecinos, j].sum())
            razon = f"{cuantos} de sus {K} vecinos más parecidos compraron este producto."
        st.write(f"**{posicion}. {productos[j]}** — Puntaje: {scores[j]:.2f}")
        st.write(razon)

with tab2:
    # TODO 6: tabla de comparación (hit rate@3 y cobertura de cada método) calculada sobre los clientes del grupo
    #         CONTROL que probaron algo nuevo (NUEVO.sum(axis=1) > 0 y es_control), con al menos un control
    #         interactivo (por ejemplo, un slider con cuántos clientes evaluar o el K del filtrado por cliente).
    #         Pista: es la misma lógica de hit_rate_at_k del notebook.
    elegibles = np.where(es_control & (NUEVO.sum(axis=1) > 0))[0]
    if len(elegibles) == 0:
        st.warning("No hay clientes elegibles del grupo control para evaluar.")
    else:
        max_clientes = len(elegibles)
        cantidad = st.slider("Clientes de control a evaluar", 1, max_clientes,
                             min(200, max_clientes), 1)
        seleccion = elegibles[:cantidad]
        resultados = []
        for nombre in ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"]:
            aciertos = []
            productos_recomendados = set()
            for i in seleccion:
                punt = puntajes(nombre, X[i], K, int(i))
                recomendaciones = top3(punt, X[i])
                aciertos.append(bool(NUEVO[i, recomendaciones].any()))
                productos_recomendados.update(recomendaciones.tolist())
            resultados.append({
                "Método": nombre,
                "Hit rate@3 (%)": round(100 * np.mean(aciertos), 2),
                "Cobertura (de 28)": len(productos_recomendados),
            })
        st.dataframe(pd.DataFrame(resultados).set_index("Método"), use_container_width=True)
        st.caption("El hit rate indica qué porcentaje compró al menos uno de los 3 productos recomendados. "
                   "La cobertura cuenta cuántos productos diferentes se recomendaron. "
                   "Se evalúan únicamente clientes del grupo control que probaron algo nuevo.")
