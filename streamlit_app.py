import streamlit as st
import subprocess, sys, os, pandas as pd

st.set_page_config(page_title='AI Surveillance Dashboard', layout='wide')
st.title('AI-Powered Surveillance (Loitering, Abandonment, Speed Anomaly)')
st.caption('Starter app. Uses YOLOv5 if available, else OpenCV fallback.')

video = st.text_input('Video file path', 'data/samples/demo_synth_abandon.mp4')
show = st.checkbox('Show realtime window (desktop only)', value=False)
save = st.checkbox('Save annotated MP4 and alerts CSV', value=True)

if st.button('Run Detection'):
    cmd = [sys.executable, 'src/main.py', '--video', video]
    if show: cmd.append('--show')
    if save: cmd.append('--save')
    with st.spinner('Processing video...'):
        p = subprocess.run(cmd, capture_output=True, text=True)
    st.write('Process finished with code', p.returncode)
    if p.stdout: st.text('STDOUT:\n' + p.stdout[-1000:])
    if p.stderr: st.text('STDERR:\n' + p.stderr[-1000:])
    if os.path.exists('outputs/alerts.csv'):
        df = pd.read_csv('outputs/alerts.csv')
        st.subheader('Alerts')
        st.dataframe(df)
    if os.path.exists('outputs/annotated.mp4'):
        st.video('outputs/annotated.mp4')
