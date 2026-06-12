import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data import (
    DIST, NAMES, COORDS, DEFAULT_DEMAND, PRODUCT_SPLIT, CAPACITY,
    CANTON_RADIUS, CANTON_COLOR,
)
from solver import solve_cvrp, extract_routes

st.set_page_config(page_title="Florida Bebidas · Cartago CVRP", page_icon="🚚", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem;}
    .stMetric {background-color: #f0f4f8; border-radius: 10px; padding: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚚 Florida Bebidas — Ruteo de Distribución, Provincia de Cartago")
st.caption(
    "Modelo de optimización (MILP) para minimizar los kilómetros recorridos por la flota, "
    f"con camiones de **{CAPACITY} pallets** de capacidad. CD = Centro de Distribución (nodo 0)."
)

# ----------------------------------------------------------------------------
# Sidebar — parámetros de demanda
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros de demanda")
st.sidebar.markdown(
    "Ajusta la **demanda semanal total (pallets)** de cada cantón. "
    "El split de productos es Imperial 50% · Pilsen 25% · Tropical 25%."
)

if st.sidebar.button("↺ Restaurar valores por defecto"):
    for i in range(1, 9):
        st.session_state[f"demand_{i}"] = DEFAULT_DEMAND[i]

demands = {}
for i in range(1, 9):
    key = f"demand_{i}"
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_DEMAND[i]
    demands[i] = st.sidebar.number_input(
        NAMES[i], min_value=0, max_value=500, step=1, key=key
    )

# ----------------------------------------------------------------------------
# Tabla de demanda por producto
# ----------------------------------------------------------------------------
st.subheader("📊 Demanda por cantón y producto (pallets/semana)")

rows = []
for i in range(1, 9):
    d = demands[i]
    rows.append(
        {
            "Cantón": NAMES[i],
            "Imperial": round(d * PRODUCT_SPLIT["Imperial"]),
            "Pilsen": round(d * PRODUCT_SPLIT["Pilsen"]),
            "Tropical": round(d * PRODUCT_SPLIT["Tropical"]),
            "Demanda total": d,
        }
    )

df = pd.DataFrame(rows)
total_row = {
    "Cantón": "TOTAL",
    "Imperial": int(df["Imperial"].sum()),
    "Pilsen": int(df["Pilsen"].sum()),
    "Tropical": int(df["Tropical"].sum()),
    "Demanda total": int(df["Demanda total"].sum()),
}
df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
st.dataframe(df, use_container_width=True, hide_index=True)

total_demand = int(sum(demands.values()))
camiones_min = int(np.ceil(total_demand / CAPACITY)) if total_demand > 0 else 0

c1, c2, c3 = st.columns(3)
c1.metric("Demanda total", f"{total_demand} pallets")
c2.metric("Capacidad por camión", f"{CAPACITY} pallets")
c3.metric("Camiones mínimos teóricos", f"⌈{total_demand}/{CAPACITY}⌉ = {camiones_min}")

# ----------------------------------------------------------------------------
# Modelo MILP
# ----------------------------------------------------------------------------
with st.expander("📐 Ver formulación del modelo (MILP)"):
    st.markdown(
        r"""
**Variables**
- $x_{ij} \in \mathbb{Z}_{\ge 0}$ : número de veces que un camión recorre el arco $i \to j$
- $y_{ij} \ge 0$ : pallets transportados sobre el arco $i \to j$

**Restricciones**
1. **Conservación de camiones** — cada cantón recibe tantos camiones como envía:
   $$\sum_j x_{kj} - \sum_i x_{ik} = 0 \quad \forall k$$
2. **Conservación de pallets / demanda** — la carga que entra menos la que sale equivale a la demanda entregada:
   $$\sum_i y_{ik} - \sum_j y_{kj} = d_k \quad \forall k \neq 0$$
   (en el CD, el flujo neto que sale equivale a la demanda total de la provincia)
3. **Capacidad / Gran M** — un camión solo puede llevar carga si transita el arco, y como máximo 24 pallets por viaje:
   $$y_{ij} \le 24 \cdot x_{ij} \quad \forall i \neq j$$

**Objetivo**
$$\min \sum_{i \neq j} d_{ij} \cdot x_{ij}$$
        """
    )

run = st.button("🧮 Optimizar rutas", type="primary", use_container_width=True)

if run:
    demand_vec = [0] + [demands[i] for i in range(1, 9)]

    if total_demand == 0:
        st.warning("La demanda total es 0 — no hay nada que distribuir.")
    else:
        try:
            with st.spinner("Resolviendo el modelo MILP con HiGHS… (puede tardar unos segundos)"):
                x_dict, total_km, info = solve_cvrp(DIST, demand_vec, capacity=CAPACITY)
                routes = extract_routes(x_dict, 9)
        except Exception as e:
            st.error(f"No se pudo resolver el modelo: {e}")
            st.stop()

        if info["status"] == 0 and info["gap"] == 0:
            st.success(
                f"✅ **Óptimo garantizado** (gap = 0%): **{total_km:.1f} km** "
                f"repartidos en **{len(routes)} viajes**."
            )
        else:
            st.warning(
                f"⚠️ Mejor solución encontrada: **{total_km:.1f} km** en **{len(routes)} viajes** "
                f"(gap = {info['gap']*100:.2f}%, status: {info['message']})."
            )

        # ------------------------------------------------------------------
        # Tabla de viajes
        # ------------------------------------------------------------------
        route_rows = []
        for idx, r in enumerate(routes, 1):
            path = " → ".join("CD" if n == 0 else NAMES[n] for n in r)
            km = sum(DIST[r[k]][r[k + 1]] for k in range(len(r) - 1))
            route_rows.append({"Viaje #": idx, "Ruta": path, "Distancia (km)": km})

        st.subheader("🛣️ Detalle de viajes óptimos")
        st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

        # ------------------------------------------------------------------
        # Mapa (matplotlib) — provincia de Cartago estilizada
        # ------------------------------------------------------------------
        st.subheader("🗺️ Mapa de rutas — Provincia de Cartago")
        st.caption("En rojo: las rutas óptimas que minimizan los km totales. Cada ruta sale y vuelve al CD.")

        def make_blob(center, radius, seed, n_pts=16, irregularity=0.28):
            """Genera un polígono orgánico (tipo 'mancha') alrededor de un centro."""
            rng = np.random.default_rng(seed)
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            radii = radius * (1 + irregularity * (rng.random(n_pts) - 0.5))
            lon = center[1] + radii * np.cos(angles) * 1.15  # corrección de aspecto
            lat = center[0] + radii * np.sin(angles)
            return np.column_stack([lon, lat])

        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        fig.patch.set_facecolor("#eef3f7")
        ax.set_facecolor("#eef3f7")

        # cantones como "manchas" coloreadas
        for i in range(1, 9):
            poly = make_blob(COORDS[i], CANTON_RADIUS[i], seed=i)
            patch = plt.Polygon(
                poly, closed=True, facecolor=CANTON_COLOR[i],
                edgecolor="white", linewidth=1.5, alpha=0.85, zorder=1,
            )
            ax.add_patch(patch)
            ax.annotate(
                f"{NAMES[i]}\n{demands[i]} pal.",
                (COORDS[i][1], COORDS[i][0]),
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#22324a", zorder=4,
            )

        # rutas en rojo (con pequeño offset para distinguir rutas superpuestas)
        for idx, r in enumerate(routes):
            offset = (idx % 5) * 0.0018
            lons = [COORDS[n][1] + offset for n in r]
            lats = [COORDS[n][0] + offset for n in r]
            ax.plot(lons, lats, color="#d6001c", linewidth=2.2, alpha=0.8,
                    zorder=3, solid_capstyle="round")

        # depósito (CD)
        lat0, lon0 = COORDS[0]
        ax.scatter(lon0, lat0, s=300, color="#ffffff", edgecolor="#d6001c",
                   linewidth=2, marker="*", zorder=5)
        ax.annotate("CD Cartago", (lon0, lat0), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9.5,
                    fontweight="bold", color="#d6001c", zorder=5)

        ax.set_title("Rutas óptimas de distribución — Provincia de Cartago",
                      fontsize=13, fontweight="bold", color="#22324a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")
        ax.margins(0.12)

        st.pyplot(fig, use_container_width=True)

else:
    st.info("Ajusta la demanda en el panel izquierdo (opcional) y presiona **Optimizar rutas**.")
