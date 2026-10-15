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

logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
with logo_col2:
    st.image('assets/logo-uacm.png', width=280)

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
