import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Empleabilidad Argentina 2023",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Empleabilidad Argentina 2023")
st.caption("Basado en la encuesta de remuneración salarial de Sysarmy")

# ── Carga de datos ──────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo)

    # Limpiar pagos_en_dolares
    df["pagos_en_dolares"] = df["pagos_en_dolares"].fillna("No")
    df["pagos_en_dolares"].replace(
        {
            "Cobro parte del salario en dólares": "Sí",
            "Cobro todo el salario en dólares": "Sí",
            "Mi sueldo está dolarizado (pero cobro en moneda local)": "No",
        },
        inplace=True,
    )

    # Seleccionar columnas de interés
    columnas = [
        "donde_estas_trabajando",
        "ultimo_salario_mensual_o_retiro_neto_en_tu_moneda_local",
        "pagos_en_dolares",
        "tipo_de_contrato",
        "sueldo_dolarizado",
        "trabajo_de",
        "seniority",
        "me_id_extra",
        "tengo_edad",
    ]
    df = df[columnas].copy()

    # Estandarizar género
    df["me_id_extra"].replace({"Hombre Cis": "Hombre", "Mujer Cis": "Mujer"}, inplace=True)

    # Renombrar columnas
    df.rename(
        columns={
            "donde_estas_trabajando": "Provincia",
            "ultimo_salario_mensual_o_retiro_neto_en_tu_moneda_local": "Salario neto",
            "pagos_en_dolares": "Pago en dólares",
            "tipo_de_contrato": "Tipo de contrato",
            "sueldo_dolarizado": "Sueldo dolarizado",
            "trabajo_de": "Profesión",
            "seniority": "Seniority",
            "me_id_extra": "Género",
            "tengo_edad": "Edad",
        },
        inplace=True,
    )

    # Convertir salario a numérico
    df["Salario neto"] = pd.to_numeric(df["Salario neto"], errors="coerce")

    return df


# ── Carga del archivo ───────────────────────────────────────────────────────
st.sidebar.header("Datos")
uploaded = st.sidebar.file_uploader(
    "Subí el CSV de Sysarmy", type=["csv"], help="Encuesta de remuneración salarial Argentina 2023"
)

if uploaded is not None:
    df = cargar_datos(uploaded)
else:
    st.info(
        "👆 Subí el archivo CSV desde el panel lateral para comenzar.\n\n"
        "El archivo esperado es: `2023.1_Sysarmy_Encuesta de remuneración salarial Argentina.csv`"
    )
    st.stop()

# ── Sidebar: Filtros ────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("Filtros")

provincias = ["Todas"] + sorted(df["Provincia"].dropna().unique().tolist())
provincia_sel = st.sidebar.selectbox("Provincia", provincias)

seniority_opts = sorted(df["Seniority"].dropna().unique().tolist())
seniority_sel = st.sidebar.multiselect("Seniority", seniority_opts, default=seniority_opts)

genero_opts = sorted(df["Género"].dropna().unique().tolist())
genero_sel = st.sidebar.multiselect("Género", genero_opts, default=genero_opts)

contrato_opts = sorted(df["Tipo de contrato"].dropna().unique().tolist())
contrato_sel = st.sidebar.multiselect("Tipo de contrato", contrato_opts, default=contrato_opts)

# Aplicar filtros
filtrado = df.copy()
if provincia_sel != "Todas":
    filtrado = filtrado[filtrado["Provincia"] == provincia_sel]
if seniority_sel:
    filtrado = filtrado[filtrado["Seniority"].isin(seniority_sel)]
if genero_sel:
    filtrado = filtrado[filtrado["Género"].isin(genero_sel)]
if contrato_sel:
    filtrado = filtrado[filtrado["Tipo de contrato"].isin(contrato_sel)]

if filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados. Ajustá los filtros del panel lateral.")
    st.stop()

# ── Métricas principales ────────────────────────────────────────────────────
st.subheader("Resumen general")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Respuestas", f"{len(filtrado):,}")
col2.metric("Salario neto promedio", f"${filtrado['Salario neto'].mean():,.0f}")
col3.metric("Salario neto mediano", f"${filtrado['Salario neto'].median():,.0f}")
pct_usd = (filtrado["Pago en dólares"] == "Sí").mean() * 100
col4.metric("Cobran en dólares", f"{pct_usd:.1f}%")

st.divider()

# ── Gráfico 1: Distribución de pagos en dólares ────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribución de pagos en dólares")
    pagos = filtrado["Pago en dólares"].value_counts()
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.pie(
        pagos,
        labels=pagos.index,
        colors=["#4c9be8", "#f28b6e"],
        autopct="%1.1f%%",
        explode=[0.05] + [0] * (len(pagos) - 1),
        shadow=True,
    )
    ax1.legend(
        labels=[f"{idx} ({val:,})" for idx, val in zip(pagos.index, pagos.values)],
        title="Pagos en dólares",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    ax1.set_title("Pagos en dólares")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

# ── Gráfico 2: Top 10 provincias por salario promedio ─────────────────────
with col_b:
    st.subheader("Top 10 provincias por salario promedio")
    provincias_top = (
        filtrado.groupby("Provincia")["Salario neto"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)
    )
    sns.set_theme(style="whitegrid")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("viridis", len(provincias_top))
    bars = ax2.barh(provincias_top.index, provincias_top.values, color=colors)
    ax2.set_xlabel("Salario promedio (ARS)")
    ax2.set_title("Provincias con mayor salario promedio")
    for bar in bars:
        w = bar.get_width()
        ax2.text(
            w + max(provincias_top.values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"${w:,.0f}",
            va="center",
            fontsize=8,
        )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

# ── Gráfico 3: Top 10 profesiones Freelance ───────────────────────────────
st.subheader("Top 10 profesiones en contratos Freelance")
freelance = filtrado[filtrado["Tipo de contrato"] == "Freelance"]["Profesión"].value_counts()

if freelance.empty:
    st.info("No hay datos de contratos Freelance con los filtros actuales.")
else:
    top10 = freelance.head(10).sort_values()
    total_freelance = freelance.sum()
    pct = (top10 / total_freelance) * 100

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    colors3 = sns.color_palette("mako", len(top10))
    bars3 = ax3.barh(top10.index, top10.values, color=colors3)
    ax3.set_title("Top 10 profesiones (contrato: Freelance)")
    ax3.set_xlabel("Cantidad de respuestas")
    gap = max(top10.values) * 0.01
    for bar, val, p in zip(bars3, top10.values, pct):
        ax3.text(
            val + gap,
            bar.get_y() + bar.get_height() / 2,
            f"{int(val):,} ({p:.1f}%)",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

st.divider()

# ── Proyección salarial 2028 ───────────────────────────────────────────────
st.subheader("Proyección salarial 2028")

col_p1, col_p2 = st.columns([1, 2])

with col_p1:
    crecimiento_anual = st.slider(
        "Crecimiento anual estimado (%)",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Ajustá la hipótesis de crecimiento anual del salario",
    ) / 100

    salario_base = filtrado["Salario neto"].mean()
    salario_2028_simple = salario_base * ((1 + crecimiento_anual) ** 5)

    # Modelo polinómico
    anios = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
    salarios_hist = salario_base / ((1 + crecimiento_anual) ** (2023 - anios.flatten()))
    salarios_hist = salarios_hist.astype(float)
    poly = PolynomialFeatures(degree=2)
    anios_poly = poly.fit_transform(anios)
    modelo_poly = LinearRegression()
    modelo_poly.fit(anios_poly, salarios_hist)
    salario_2028_poly = modelo_poly.predict(poly.transform(np.array([[2028]])))[0]

    st.metric("Salario base 2023 (filtrado)", f"${salario_base:,.0f}")
    st.metric("Proyección 2028 (interés compuesto)", f"${salario_2028_simple:,.0f}")
    st.metric("Proyección 2028 (modelo polinómico)", f"${salario_2028_poly:,.0f}")

with col_p2:
    anios_plot = list(range(2018, 2029))
    valores_plot = [salario_base / ((1 + crecimiento_anual) ** (2023 - a)) for a in range(2018, 2024)]
    valores_plot += [salario_base * ((1 + crecimiento_anual) ** a) for a in range(1, 6)]

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(anios_plot[:6], valores_plot[:6], "o-", color="#4c9be8", label="Histórico estimado")
    ax4.plot(anios_plot[5:], valores_plot[5:], "o--", color="#f28b6e", label="Proyección 2028")
    ax4.axvline(x=2023, color="gray", linestyle=":", linewidth=1)
    ax4.set_xlabel("Año")
    ax4.set_ylabel("Salario neto promedio (ARS)")
    ax4.set_title(f"Evolución y proyección salarial (crecimiento {crecimiento_anual*100:.0f}% anual)")
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.legend()
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.divider()

# ── Tabla exploratoria ─────────────────────────────────────────────────────
with st.expander("📋 Ver datos filtrados"):
    st.dataframe(
        filtrado.reset_index(drop=True),
        use_container_width=True,
        height=300,
    )
    st.caption(f"{len(filtrado):,} filas · {filtrado.shape[1]} columnas")
