import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
FEATURES = ['device_type','flow_rate_ml_min','inlet_diameter_mm','hydraulic_diameter_mm','vol_avg_shear_rate','max_wall_shear_rate','p95_wall_shear_rate','throat_avg_shear_rate','vol_avg_exposure_time','max_exposure_time','throat_exposure_time','vol_physics_index','throat_physics_index','max_physics_index']
PHYSICS = ['vol_physics_index','throat_physics_index','max_physics_index']
DEVICES = ['baseline','herringbone','reduced_gap','reducedport']
@st.cache_resource
def model():
    return joblib.load(ROOT / 'model.joblib')
@st.cache_data
def dataset():
    return pd.read_csv(ROOT / 'reference_data.csv')

st.set_page_config(page_title='HemoPredict AI',page_icon='🩸',layout='wide')
st.markdown('''<style>.stApp{background:#081323;color:#f0f6ff}.block-container{padding-top:2rem;max-width:1250px}h1,h2,h3{color:#f4f9ff}div[data-testid="stMetric"]{background:#12243c;padding:17px;border:1px solid #294562;border-radius:14px}div.stButton>button{background:#00bfe8;color:#001827;font-weight:700;border-radius:10px}</style>''',unsafe_allow_html=True)
st.caption('ALIBABA CLOUD AI HACKATHON 2026  |  RESEARCH PROTOTYPE  |  MUHAMMAD SAAD HANIF')
st.title('🩸 HemoPredict AI')
st.markdown('### AI-Based Blood Damage Prediction Model for Internal Artificial Organs')
st.markdown('**CFD-informed machine learning for early-stage blood-contacting device research**')
st.info('Research prototype only. Predictions are log10(HI) estimates, not measured clinical hemolysis. CFD-derived shear and exposure inputs are required. Not validated for unseen device geometries.')

ref=dataset()
left,right=st.columns([1.05,1])
with left:
    st.subheader('01 / Input conditions')
    mode=st.radio('Input method',['Verified dataset example','Enter all CFD parameters'],horizontal=True)
    if mode=='Verified dataset example':
        device=st.selectbox('Device configuration',DEVICES)
        available=ref.loc[ref.device_type==device].copy()
        flow=st.selectbox('Available flow rate (mL/min)',sorted(available.flow_rate_ml_min.unique().tolist()),index=min(5,len(available.flow_rate_ml_min.unique())-1))
        chosen=available.loc[available.flow_rate_ml_min==flow].iloc[0]
        row={f:chosen[f] for f in FEATURES}
        st.caption('Inputs loaded from an existing CFD-derived dataset row. This is a demonstration, NOT independent external validation.')
        with st.expander('See exact model inputs'):
            st.dataframe(pd.DataFrame([row]).T.rename(columns={0:'Value'}),use_container_width=True)
    else:
        st.warning('Enter all measured/simulated CFD quantities. The app cannot infer shear and exposure from geometry and flow alone.')
        device=st.selectbox('Device configuration',DEVICES)
        row={'device_type':device}
        a,b=st.columns(2)
        for i,f in enumerate(FEATURES[1:]):
            default=float(ref.loc[ref.device_type==device,f].median())
            with (a if i%2==0 else b):
                row[f]=st.number_input(f.replace('_',' '),min_value=0.0,value=default,format='%.12g',help='CFD-derived input; preserve original dataset units')
    go=st.button('Run real model prediction',type='primary',use_container_width=True)
with right:
    st.subheader('02 / Prediction')
    if go:
        x=pd.DataFrame([row],columns=FEATURES)
        prediction=float(model().predict(x)[0])
        st.metric('Predicted log10(HI)',f'{prediction:.5f}')
        st.caption('Computed live by the saved trained Extra Trees pipeline. No hardcoded prediction.')
        if mode=='Verified dataset example':
            actual=float(chosen['target_log10_hi'])
            c1,c2=st.columns(2)
            c1.metric('Dataset target',f'{actual:.5f}')
            c2.metric('Absolute error',f'{abs(prediction-actual):.5f}')
            st.warning('This row may have been used during model fitting. Its error is illustrative, not a test-set score.')
        if any(row[f] < float(ref[f].min()) or row[f] > float(ref[f].max()) for f in FEATURES[1:]):
            st.error('One or more numeric inputs are outside the dataset range: extrapolation risk.')
        if mode=='Enter all CFD parameters':
            expected=[row['vol_avg_shear_rate']**2*row['vol_avg_exposure_time'],row['throat_avg_shear_rate']**2*row['throat_exposure_time'],row['max_wall_shear_rate']**2*row['max_exposure_time']]
            if any(abs(row[f]-e)>max(abs(e)*0.15,1e-12) for f,e in zip(PHYSICS,expected)):
                st.warning('Physics-index inputs differ from approximate shear² × exposure relationships. Verify the source CFD and index definitions.')
    else:
        st.info('Choose a verified example or enter complete CFD inputs, then run prediction.')
    st.subheader('03 / Verified evaluation')
    m1,m2,m3=st.columns(3)
    m1.metric('Held-out R²','0.9818')
    m2.metric('Held-out MAE','0.1201')
    m3.metric('Held-out RMSE','0.3172')
    st.caption('Development holdout: 94 samples, seen device families; not an untouched external test. Target units: log10(HI).')

st.divider()
st.subheader('04 / Evidence & limitations')
t1,t2,t3=st.tabs(['Prediction diagnostics','Feature importance','Unseen geometry'])
with t1:
    st.image(str(ROOT/'prediction_diagnostics.png'),use_container_width=True)
with t2:
    st.image(str(ROOT/'permutation_importance.png'),use_container_width=True)
with t3:
    metrics=pd.read_csv(ROOT/'unseen_device_metrics.csv')
    st.dataframe(metrics,use_container_width=True,hide_index=True)
    st.caption('Leave-one-device-out tests: whole geometry held out during training. Generalization is weaker than seen-family holdout performance.')
st.caption('Developed by Muhammad Saad Hanif | FYP research prototype | Source: FYP physics-informed dataset, inspired by Micromachines study | Hosted via Streamlit, not Alibaba Cloud.')
