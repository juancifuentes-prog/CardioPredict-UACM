import io, json
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='CardioPredict | UACM · Prototipo',page_icon='🫀',layout='centered')

st.markdown(
    '''
    <style>
    .block-container{max-width:850px;padding-top:2rem}
    .hero{padding:1.4rem;border-radius:16px;background:#162a4a;color:white;margin-bottom:1rem;text-align:center}
    .hero h1{color:white;margin:0;font-size:2rem}
    .hero p{margin-bottom:0;color:#dbe7f3}
    .institution{text-align:center;color:#4b5563;font-size:.95rem;margin:.35rem 0 1rem 0;line-height:1.45}
    .sub{font-size:.9rem;color:#566}
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style="text-align:center; margin:0 auto 0.5rem auto;">
        <img src="data:image/jpeg;base64,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"
             alt="Logo de la Universidad Autónoma de la Ciudad de México"
             style="width:260px; max-width:70%; height:auto;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="institution">
        <strong>Universidad Autónoma de la Ciudad de México</strong><br>
        Academia de Nutrición y Salud · Plantel Casa Libertad
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero"><h1>CardioPredict</h1><p>Modelo antropométrico de alteraciones aterogénicas · Prototipo académico</p></div>',
    unsafe_allow_html=True
)

st.warning('Modelo exploratorio (n = 43). No constituye diagnóstico ni estima eventos cardiovasculares. No utilizar para decisiones clínicas.')

@st.cache_resource
def cargar_modelo():
    return joblib.load('cardiopredict_modelo.joblib')

@st.cache_data
def cargar_metricas():
    with open('cardiopredict_metricas.json', encoding='utf-8') as f:
        return json.load(f)

modelo=cargar_modelo()
metricas=cargar_metricas()

with st.form('formulario'):
    st.subheader('Datos antropométricos')
    a,b=st.columns(2)
    with a:
        imc=st.number_input('IMC (kg/m²)',min_value=10.0,max_value=75.0,value=25.0,step=0.1)
        cintura=st.number_input('Circunferencia de cintura (cm)',min_value=40.0,max_value=200.0,value=85.0,step=0.1)
    with b:
        cuello=st.number_input('Circunferencia de cuello (cm)',min_value=15.0,max_value=70.0,value=35.0,step=0.1)
        estatura=st.number_input('Estatura (cm)',min_value=100.0,max_value=230.0,value=170.0,step=0.1)
    ratio=cintura/estatura
    st.caption(f'Relación cintura/talla calculada automáticamente: {ratio:.3f} (cm/cm)')
    enviado=st.form_submit_button('Calcular probabilidad',type='primary',use_container_width=True)

if enviado:
    datos=pd.DataFrame([{'IMC':imc,'Cintura_cm':cintura,'Cuello_cm':cuello,'Cintura_Talla':ratio}])
    p=float(modelo.predict_proba(datos)[0,1])
    st.divider()
    st.subheader('Resultado del modelo')
    st.metric('Probabilidad estimada de ≥2 alteraciones aterogénicas',f'{p*100:.1f} %')
    st.progress(min(max(p,0),1))
    st.caption('Desenlace: presencia de al menos 2 entre hipertrigliceridemia, HDL bajo y LDL alto. La probabilidad no es equivalente al riesgo de infarto ni al diagnóstico de enfermedad cardiovascular.')
    st.info('Interpretación exploratoria: la probabilidad surge de una regresión logística entrenada con la muestra disponible. No existe todavía un punto de corte clínico validado.')
    report=pd.DataFrame([{
        'fecha_calculo':datetime.now().strftime('%Y-%m-%d %H:%M'),
        'IMC_kg_m2':imc,
        'cintura_cm':cintura,
        'cuello_cm':cuello,
        'estatura_cm':estatura,
        'cintura_talla':round(ratio,4),
        'probabilidad_2_o_mas_alteraciones':round(p,5),
        'version_modelo':'exploratorio-1',
        'uso':'investigacion_no_diagnostico'
    }])
    st.download_button(
        'Descargar reporte CSV (sin datos identificatorios)',
        data=report.to_csv(index=False).encode('utf-8-sig'),
        file_name='cardiopredict_reporte.csv',
        mime='text/csv'
    )

with st.expander('Desarrollo, validación y límites del modelo'):
    st.write(f"**Muestra de entrenamiento:** {metricas['n']} participantes ({metricas['positivos']} con desenlace, {metricas['negativos']} sin desenlace).")
    st.write(f"**AUC-ROC por validación cruzada estratificada de {metricas['cv_folds']} pliegues:** {metricas['roc_auc_cv']:.3f}")
    st.write(f"**Brier en validación cruzada:** {metricas['brier_cv']:.3f}")
    st.write('Variables: IMC, cintura, cuello y relación cintura/talla. Esta última depende matemáticamente de cintura y estatura. No existen aquí validaciones externas ni umbrales diagnósticos calibrados.')
    st.write('La interfaz solo procesa las entradas de esta consulta; no incluye una base de pacientes. Si se despliega en un servidor externo, debe revisarse la política de privacidad y la seguridad del alojamiento.')

st.caption('Prototipo de investigación académica · UACM · Modelo 1 · Requiere ampliación de muestra, evaluación de calibración y validación externa.')
