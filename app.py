import json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CardioPredict-UACM · v2.1",
    page_icon="🫀",
    layout="wide"
)

DEFAULTS = {
    "imc": 25.0,
    "cintura": 85.0,
    "cuello": 34.0,
    "grasa_bia": 35.0,
    "grasa_visceral": 1.9,
    "masa_muscular": 19.5,
    "angulo_fase": 5.6,
    "fuerza": 15.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("resultado", None)

def limpiar_campos():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["resultado"] = None

@st.cache_resource
def cargar_modelo():
    return joblib.load("cardiopredict_v2_1_modelo.joblib")

@st.cache_data
def cargar_metricas():
    with open("cardiopredict_v2_1_metricas.json", encoding="utf-8") as f:
        return json.load(f)

modelo = cargar_modelo()
metricas = cargar_metricas()

st.markdown(
    """
    <style>
    :root{
        --navy:#0b2d56;
        --navy2:#173f73;
        --maroon:#8f1230;
        --maroon2:#a8173b;
        --ink:#14213d;
        --muted:#667085;
        --line:#d9e1ea;
        --soft:#f5f8fb;
    }
    .block-container{
        max-width:1240px;
        padding-top:1.1rem;
        padding-bottom:2rem;
    }
    header[data-testid="stHeader"]{background:transparent;}
    .brand-wrap{
        display:grid;
        grid-template-columns:minmax(150px,0.9fr) minmax(340px,2.5fr) minmax(150px,0.9fr);
        gap:24px;
        align-items:center;
        padding:18px 24px 16px;
        border:1px solid #e5e9ef;
        border-radius:24px;
        background:white;
        box-shadow:0 10px 30px rgba(11,45,86,.08);
        margin-bottom:12px;
    }
    .brand-logo{display:flex;align-items:center;justify-content:center;min-width:0;}
    .brand-logo img{max-width:100%;height:auto;object-fit:contain;}
    .brand-logo.uacm img{width:230px;max-height:122px;}
    .brand-logo.lab img{width:170px;max-height:150px;border-radius:8px;}
    .brand-center{text-align:center;min-width:0;padding:0 8px;}
    .brand-title{
        font-size:clamp(2rem,4vw,3.25rem);
        line-height:1.02;
        font-weight:850;
        letter-spacing:-.035em;
        color:var(--navy);
        margin:0;
    }
    .brand-title span{color:var(--maroon);}
    .brand-sub{font-size:1.12rem;color:#253754;margin:8px 0 10px;}
    .version-pill{
        display:inline-block;
        background:linear-gradient(90deg,var(--maroon),#b21d44);
        color:white;
        font-weight:750;
        border-radius:999px;
        padding:6px 22px;
        letter-spacing:.02em;
    }
    .wave{
        height:6px;
        border-radius:99px;
        background:linear-gradient(90deg,var(--maroon),#d997a7 32%,var(--navy) 74%,#567ba9);
        margin-bottom:20px;
    }
    .intro{
        border-radius:24px;
        padding:24px 26px;
        background:
          radial-gradient(circle at 92% 20%,rgba(143,18,48,.13),transparent 28%),
          radial-gradient(circle at 76% 80%,rgba(11,45,86,.14),transparent 34%),
          linear-gradient(135deg,#f8fbff,#eef4fa);
        border:1px solid #dce5ee;
        margin-bottom:18px;
    }
    .intro h2{margin:0;color:var(--navy);font-size:1.8rem;}
    .intro p{margin:.45rem 0 0;color:#42526b;font-size:1.03rem;}
    .info-card{
        border:1px solid #dfe6ee;
        border-radius:22px;
        background:white;
        padding:22px;
        box-shadow:0 8px 24px rgba(11,45,86,.06);
        margin-bottom:18px;
    }
    .heart-panel{
        min-height:300px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        text-align:center;
        border-radius:22px;
        background:
          radial-gradient(circle at 40% 28%,rgba(255,255,255,.22),transparent 19%),
          linear-gradient(145deg,#0b2d56,#164e85 54%,#8f1230);
        color:white;
        overflow:hidden;
        position:relative;
        margin-bottom:18px;
    }
    .heart-panel:before,.heart-panel:after{
        content:"";
        position:absolute;
        border-radius:50%;
        border:2px solid rgba(255,255,255,.16);
    }
    .heart-panel:before{width:250px;height:250px;right:-70px;top:-65px;}
    .heart-panel:after{width:180px;height:180px;left:-65px;bottom:-60px;}
    .heart-icon{font-size:5rem;line-height:1;margin-bottom:.4rem;}
    .heart-panel h3{font-size:1.5rem;margin:0;z-index:1;}
    .heart-panel p{max-width:280px;color:#edf5ff;z-index:1;margin:.6rem 1.2rem 0;}
    .model-stat{
        display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:16px;
    }
    .stat-box{padding:12px;border-radius:14px;background:#f4f7fb;border:1px solid #e2e8f0;text-align:center;}
    .stat-box b{display:block;color:var(--navy);font-size:1.2rem;}
    div[data-testid="stForm"]{
        border:1px solid #dbe4ee !important;
        border-radius:24px !important;
        padding:22px 24px 18px !important;
        background:white;
        box-shadow:0 10px 28px rgba(11,45,86,.07);
    }
    div[data-testid="stForm"] h3{color:var(--navy);}
    div[data-baseweb="input"]{
        border-radius:12px !important;
    }
    .stButton>button, div[data-testid="stFormSubmitButton"]>button{
        border-radius:12px !important;
        min-height:48px;
        font-weight:750 !important;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]{
        background:linear-gradient(90deg,var(--maroon),var(--maroon2)) !important;
        border-color:var(--maroon) !important;
        color:white !important;
    }
    .result-card{
        border:1px solid #dbe4ee;
        border-radius:22px;
        background:linear-gradient(135deg,#f8fbff,#edf5ff);
        padding:20px 22px;
        margin-top:16px;
    }
    .result-head{color:var(--navy);font-size:1.35rem;font-weight:800;margin-bottom:8px;}
    .result-grid{display:grid;grid-template-columns:1fr 145px;gap:20px;align-items:center;}
    .result-value{font-size:3.4rem;font-weight:900;letter-spacing:-.04em;color:var(--navy);}
    .result-label{font-size:1.03rem;color:#344054;margin-bottom:6px;}
    .donut{
        width:125px;height:125px;border-radius:50%;
        display:grid;place-items:center;margin:auto;
        background:conic-gradient(var(--maroon) 0 var(--pct),#dfe8f1 var(--pct) 100%);
        position:relative;
    }
    .donut:after{
        content:"";width:88px;height:88px;border-radius:50%;background:white;position:absolute;
    }
    .donut span{position:relative;z-index:1;font-size:1.9rem;font-weight:850;color:var(--navy);}
    .classification{
        margin-top:14px;padding:12px 14px;border-radius:14px;
        background:white;border-left:5px solid var(--maroon);color:#344054;
    }
    .placeholder{
        border:1px dashed #b8c6d6;border-radius:20px;padding:22px;
        background:#f8fbff;color:#667085;margin-top:16px;
    }
    .disclaimer{
        margin-top:20px;padding:14px 18px;border-radius:15px;
        background:#f1f4f7;color:#475467;text-align:center;border:1px solid #e1e6eb;
    }
    @media (max-width:900px){
        .brand-wrap{grid-template-columns:1fr;gap:10px;}
        .brand-logo.uacm img{width:190px;}
        .brand-logo.lab img{width:135px;}
        .result-grid{grid-template-columns:1fr;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="brand-wrap">
      <div class="brand-logo uacm">
        <img src="data:image/jpeg;base64,{uacm}" alt="Logo UACM">
      </div>
      <div class="brand-center">
        <div class="brand-title">CardioPredict-<span>UACM</span></div>
        <div class="brand-sub">Modelo exploratorio de carga aterogénica</div>
        <div class="version-pill">Versión 2.1 · n = {metricas['n_total_analyzed']}</div>
      </div>
      <div class="brand-logo lab">
        <img src="data:image/jpeg;base64,{cifuentes}" alt="Logo Cifuentes Lab">
      </div>
    </div>
    <div class="wave"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="intro">
      <h2>Estimación exploratoria de carga aterogénica</h2>
      <p>Ingresa los datos antropométricos, de bioimpedancia y fuerza para estimar la probabilidad de presentar ≥2 alteraciones aterogénicas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.82, 2.18], gap="large")

with left:
    st.markdown(
        """
        <div class="heart-panel">
          <div class="heart-icon">🫀</div>
          <h3>Perfil cardiometabólico</h3>
          <p>Herramienta académica para explorar patrones asociados con carga aterogénica en la muestra estudiada.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m = metricas["metrics_oof_nested"]
    st.markdown(
        f"""
        <div class="info-card">
          <h3 style="margin-top:0;color:#0b2d56;">Acerca del modelo</h3>
          <p style="color:#475467;">Regresión logística penalizada L2. El desenlace corresponde a la presencia de <b>2 o más alteraciones</b> entre siete marcadores aterogénicos.</p>
          <div class="model-stat">
            <div class="stat-box"><b>{metricas['n_total_analyzed']}</b><span>participantes</span></div>
            <div class="stat-box"><b>{m['auc']:.3f}</b><span>AUC-ROC</span></div>
            <div class="stat-box"><b>{m['sensitivity']*100:.1f}%</b><span>sensibilidad</span></div>
            <div class="stat-box"><b>{m['specificity']*100:.1f}%</b><span>especificidad</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("### 🧮 Calculadora de riesgo aterogénico")
    st.caption("Las ocho variables corresponden al modelo v2.1. Edad y respuesta a bicicleta no se incluyen porque no mejoraron el desempeño interno.")

    with st.form("formulario_v2_1", clear_on_submit=False):
        r1 = st.columns(4)
        with r1[0]:
            st.number_input("IMC (kg/m²)", min_value=10.0, max_value=70.0, step=0.1, key="imc")
        with r1[1]:
            st.number_input("Circunferencia de cintura (cm)", min_value=40.0, max_value=180.0, step=0.1, key="cintura")
        with r1[2]:
            st.number_input("Circunferencia de cuello (cm)", min_value=20.0, max_value=60.0, step=0.1, key="cuello")
        with r1[3]:
            st.number_input("Grasa corporal por BIA (%)", min_value=1.0, max_value=70.0, step=0.1, key="grasa_bia")

        r2 = st.columns(4)
        with r2[0]:
            st.number_input("Grasa visceral (L)", min_value=0.1, max_value=10.0, step=0.1, key="grasa_visceral")
        with r2[1]:
            st.number_input("Masa musculoesquelética (kg)", min_value=5.0, max_value=60.0, step=0.1, key="masa_muscular")
        with r2[2]:
            st.number_input("Ángulo de fase (°)", min_value=2.0, max_value=10.0, step=0.1, key="angulo_fase")
        with r2[3]:
            st.number_input("Fuerza de agarre ajustada", min_value=1.0, max_value=60.0, step=0.1, key="fuerza")

        b1, b2 = st.columns([2, 1])
        with b1:
            calcular = st.form_submit_button("🧮  Calcular riesgo", type="primary", use_container_width=True)
        with b2:
            limpiar = st.form_submit_button("↻  Limpiar", use_container_width=True, on_click=limpiar_campos)

    if calcular:
        datos = np.array([[
            st.session_state.imc,
            st.session_state.cintura,
            st.session_state.cuello,
            st.session_state.grasa_bia,
            st.session_state.grasa_visceral,
            st.session_state.masa_muscular,
            st.session_state.angulo_fase,
            st.session_state.fuerza,
        ]], dtype=float)

        p = float(modelo.predict_proba(datos)[0, 1])
        st.session_state["resultado"] = {
            "p": p,
            "inputs": {
                "IMC_kg_m2": st.session_state.imc,
                "cintura_cm": st.session_state.cintura,
                "cuello_cm": st.session_state.cuello,
                "grasa_corporal_BIA_pct": st.session_state.grasa_bia,
                "grasa_visceral_L": st.session_state.grasa_visceral,
                "masa_musculoesqueletica_kg": st.session_state.masa_muscular,
                "angulo_fase_grados": st.session_state.angulo_fase,
                "fuerza_agarre_ajustada": st.session_state.fuerza,
            },
        }

    resultado = st.session_state.get("resultado")
    if resultado:
        p = resultado["p"]
        pct = p * 100
        clasificacion = (
            "Patrón compatible con ≥2 alteraciones aterogénicas."
            if p >= 0.50
            else "Patrón compatible con 0–1 alteración aterogénica."
        )
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-head">📊 Resultado del modelo</div>
              <div class="result-grid">
                <div>
                  <div class="result-label">Probabilidad estimada de carga aterogénica elevada:</div>
                  <div class="result-value">{pct:.1f}%</div>
                  <div class="classification"><b>Clasificación exploratoria:</b> {clasificacion}</div>
                </div>
                <div class="donut" style="--pct:{pct:.1f}%"><span>{pct:.0f}%</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        report = pd.DataFrame([{
            "fecha_calculo": datetime.now().strftime("%Y-%m-%d %H:%M"),
            **resultado["inputs"],
            "probabilidad_carga_aterogenica_ge2": round(p, 5),
            "version_modelo": "CardioPredict-v2.1-depurado",
            "uso": "investigacion_no_diagnostico",
        }])
        st.download_button(
            "Descargar reporte CSV",
            data=report.to_csv(index=False).encode("utf-8-sig"),
            file_name="cardiopredict_v2_1_reporte.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div class="placeholder">
              <b>📊 Resultado</b><br>
              El resultado se mostrará aquí después de calcular.
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Información metodológica y límites"):
    ci = metricas["bootstrap_95ci"]
    m = metricas["metrics_oof_nested"]
    st.write(f"**Muestra analizada:** {metricas['n_total_analyzed']} participantes.")
    st.write(f"**Distribución del desenlace:** {metricas['target_distribution']['positive']} con ≥2 alteraciones y {metricas['target_distribution']['negative']} con 0–1.")
    st.write(f"**AUC-ROC:** {m['auc']:.3f} (IC95% bootstrap {ci['auc'][0]:.3f}–{ci['auc'][1]:.3f}).")
    st.write(f"**Sensibilidad:** {m['sensitivity']*100:.1f}% · **Especificidad:** {m['specificity']*100:.1f}% · **Brier:** {m['brier']:.3f}.")
    st.write("Validación cruzada anidada estratificada: 5 pliegues externos × 10 repeticiones. El modelo es exploratorio y requiere validación externa.")

st.markdown(
    """
    <div class="disclaimer">
      🧪 <b>Uso académico y de investigación.</b> No constituye diagnóstico, no estima riesgo cardiovascular absoluto y no sustituye la evaluación clínica ni el juicio profesional.
    </div>
    """,
    unsafe_allow_html=True,
)
