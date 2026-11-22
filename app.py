import os, json, datetime as dt
import numpy as np, pandas as pd, joblib
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="PakcoyMonitor", page_icon="🥬", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(BASE, f)
HIST = P('history.csv')
IMG, WINDOW = 224, 10
FEATURES = ['moisture', 'temp', 'dap', 'maturity']

# acuan estimasi (GANTI dengan hasil pengukuran sendiri)
REF = pd.DataFrame({'dap': [10, 20, 30, 40, 50],
                    'tinggi_cm': [4, 9, 15, 20, 22],
                    'lebar_daun_cm': [2.5, 5, 8, 10.5, 11.5]})

# ---------------- histori ----------------
def log(aksi, detail=''):
    wib = dt.timezone(dt.timedelta(hours=7))
    row = pd.DataFrame([{'Waktu': dt.datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S'),
                         'Aktivitas': aksi, 'Detail': detail}])
    row.to_csv(HIST, mode='a', header=not os.path.exists(HIST), index=False)

# ---------------- load resource ----------------
@st.cache_data
def load_data():
    return pd.read_csv(P('sensor_raw.csv')), pd.read_csv(P('daily_pakcoy.csv'))

@st.cache_resource
def load_lstm():
    import tensorflow as tf
    m = tf.keras.models.load_model(P('lstm_pakcoy.keras'), compile=False)
    return m, joblib.load(P('scaler_lstm.pkl')), joblib.load(P('phase_thr.pkl'))

@st.cache_resource
def load_cnn():
    import tensorflow as tf
    m = tf.keras.models.load_model(P('cnn_pakcoy.keras'), compile=False)
    return m, json.load(open(P('class_names.json')))

def get_phase(m, thr):
    if m >= thr['Siap Panen']: return 'Siap Panen'
    if m >= thr['Mendekati Panen']: return 'Mendekati Panen'
    return 'Masa Pertumbuhan'

raw, daily = load_data()

# ---------------- navigasi ----------------
st.sidebar.title("🥬 PakcoyMonitor")
page = st.sidebar.radio("Menu", ["Dasbor", "Prediksi", "Dataset", "Histori"])
if st.session_state.get('last_page') != page:
    log('Membuka halaman', page)
    st.session_state['last_page'] = page

# ---------------- grafik live ----------------
@st.fragment(run_every=2)
def live_chart(raw):
    if 'idx' not in st.session_state: st.session_state.idx = 5
    if st.session_state.get('play', True):
        st.session_state.idx = (st.session_state.idx + 1) % len(raw)
    i = st.session_state.idx
    cur = raw.iloc[i]

    a, b, c, d = st.columns(4)
    a.metric("Kelembaban tanah", f"{cur['moisture']:.0f} %")
    b.metric("Suhu", f"{cur['temp']:.1f} °C")
    c.metric("Kondisi tanah", str(cur['soil_cond']))
    d.metric("Kematangan", f"{cur['maturity']:.0f} %")

    win = raw.iloc[max(0, i - 29):i + 1].copy()
    win['label'] = 'DAP ' + win['dap'].astype(str) + ' ' + win['time'].astype(str).str[:5]
    win = win.set_index('label')
    g1, g2 = st.columns(2)
    g1.caption("Kelembaban tanah (%)"); g1.line_chart(win[['moisture']])
    g2.caption("Suhu (°C)");            g2.line_chart(win[['temp']])

# ================= DASBOR =================
if page == "Dasbor":
    st.title("Dasbor Monitoring Pakcoy")
    st.toggle("▶ Live (simulasi data sensor)", value=True, key="play")
    live_chart(raw)

    st.divider()
    st.subheader("Estimasi tinggi & lebar daun berdasarkan usia")
    dap = st.slider("Usia tanaman (hari setelah tanam)", 10, 50, 30)
    tinggi = np.interp(dap, REF['dap'], REF['tinggi_cm'])
    lebar = np.interp(dap, REF['dap'], REF['lebar_daun_cm'])
    c1, c2 = st.columns(2)
    c1.metric("Tinggi tanaman", f"{tinggi:.1f} cm")
    c2.metric("Lebar daun", f"{lebar:.1f} cm")
    st.caption("Nilai estimasi dari tabel acuan, bukan hasil pengukuran sensor.")

# ================= PREDIKSI =================
elif page == "Prediksi":
    st.title("Prediksi")
    tab_lstm, tab_cnn = st.tabs(["📈 Prediksi LSTM (4 hari)", "🖼️ Analisis Gambar CNN"])

    # ---------- LSTM ----------
    with tab_lstm:
        min_dap = int(daily['dap'].min()) + WINDOW - 1
        max_dap = int(daily['dap'].max())
        end_dap = st.slider("DAP terakhir data input", min_dap, max_dap, max_dap)
        last10 = daily[daily['dap'] <= end_dap].tail(WINDOW)

        st.markdown("**Data sensor 10 hari terakhir (input)**")
        st.dataframe(last10[['dap', 'moisture', 'temp', 'maturity', 'phase']].round(2),
                     hide_index=True)

        if st.button("Prediksi 4 hari ke depan", type="primary"):
            model, scaler, thr = load_lstm()
            arr = scaler.transform(last10[FEATURES])
            out = model.predict(arr[None], verbose=0)[0]
            mn, mx = scaler.data_min_[3], scaler.data_max_[3]
            out = np.clip(out * (mx - mn) + mn, 0, 100)
            res = pd.DataFrame({
                'Hari': [f'+{i}' for i in range(1, 5)],
                'DAP': [end_dap + i for i in range(1, 5)],
                'Kematangan (%)': np.round(out, 1),
                'Fase': [get_phase(v, thr) for v in out]})
            st.session_state['lstm'] = {'res': res, 'hist': last10}
            log('Prediksi LSTM', f'DAP akhir {end_dap}, hasil: ' +
                ', '.join(f"{r.DAP}={r._3}%" for r in res.itertuples()))

        if 'lstm' in st.session_state:
            res, hist = st.session_state['lstm']['res'], st.session_state['lstm']['hist']
            st.markdown("**Hasil prediksi**")
            st.dataframe(res, hide_index=True)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(hist['dap'], hist['maturity'], 'o-', label='Data 10 hari')
            ax.plot(res['DAP'], res['Kematangan (%)'], 's--', label='Prediksi 4 hari')
            ax.set_xlabel('DAP'); ax.set_ylabel('Kematangan (%)'); ax.legend(); ax.grid(alpha=.3)
            st.pyplot(fig)

            cols = st.columns(4)
            for col, r in zip(cols, res.itertuples()):
                path = P(f"sample_images/{r.Fase.replace(' ', '_')}.jpg")
                if os.path.exists(path):
                    col.image(path, caption=f"{r.Hari} hari (DAP {r.DAP})\n{r.Fase} · {r._3}%")

    # ---------- CNN ----------
    with tab_cnn:
        thr_conf = st.slider("Batas keyakinan minimum", 0.50, 0.95, 0.75, 0.05)
        f = st.file_uploader("Upload foto tanaman", type=['jpg', 'jpeg', 'png', 'webp'])
        if f:
            key = f"{f.name}-{f.size}"
            if st.session_state.get('last_upload') != key:
                log('Upload gambar CNN', f.name); st.session_state['last_upload'] = key
            img = Image.open(f).convert('RGB')
            st.image(img, width=350)
            if st.button("Analisis gambar", type="primary"):
                model, names = load_cnn()
                x = np.array(img.resize((IMG, IMG), Image.BILINEAR), dtype='float32')[None]
                p = model.predict(x, verbose=0)[0]
                i = int(np.argmax(p))
                if names[i] == 'Bukan_Pakcoy' or p[i] < thr_conf:
                    st.error(f"❌ Bukan tanaman pakcoy / tidak dikenali ({p[i]*100:.1f}%)")
                    hasil = 'Bukan pakcoy'
                else:
                    st.success(f"✅ {names[i].replace('_', ' ')} ({p[i]*100:.1f}%)")
                    hasil = names[i]
                st.bar_chart(pd.DataFrame({'Probabilitas': p}, index=[n.replace('_', ' ') for n in names]))
                log('Analisis CNN', f"{f.name} → {hasil} ({p[i]*100:.1f}%)")

# ================= DATASET =================
elif page == "Dataset":
    st.title("Dataset")
    st.subheader("Data sensor")
    st.dataframe(raw, hide_index=True)
    st.subheader("Contoh gambar")
    cols = st.columns(3)
    for col, c in zip(cols, ['Masa_Pertumbuhan', 'Mendekati_Panen', 'Siap_Panen']):
        col.image(P(f'sample_images/{c}.jpg'), caption=c.replace('_', ' '))

# ================= HISTORI =================
elif page == "Histori":
    st.title("Histori Aktivitas")
    if os.path.exists(HIST):
        h = pd.read_csv(HIST).iloc[::-1]
        st.dataframe(h, hide_index=True)
        st.download_button("Unduh CSV", h.to_csv(index=False), "histori.csv")
        if st.button("Hapus histori"):
            os.remove(HIST); log('Menghapus histori'); st.rerun()
    else:
        st.info("Belum ada aktivitas.")
