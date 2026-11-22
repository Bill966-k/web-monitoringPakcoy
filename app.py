import os, json, datetime as dt
import numpy as np, pandas as pd, joblib
import streamlit as st
import altair as alt
from PIL import Image

st.set_page_config(page_title="PakcoyLab", page_icon="🌱", layout="wide",
                   initial_sidebar_state="collapsed")

BASE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(BASE, f)
HIST = P('history.csv')
IMG, WINDOW = 224, 10
FEATURES = ['moisture', 'temp', 'dap', 'maturity']
LIME, MINT, AMBER, MUTED, GRID = '#B8F26B', '#5EEAD4', '#FBBF24', '#8FA396', '#22302A'
PHASE_COLOR = {'Masa Pertumbuhan': MINT, 'Mendekati Panen': AMBER, 'Siap Panen': LIME}
PAGES = ['Beranda', 'Sensor Live', 'Prediksi', 'Dataset', 'Histori', 'Pengaturan']

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body, .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: radial-gradient(1100px 480px at 8% -8%, #1d3326 0%, #0D1411 55%); }
header[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], footer { display: none; }
.block-container { padding-top: 1.2rem; max-width: 1180px; }
.topbar { display:flex; justify-content:space-between; align-items:center;
  padding:14px 20px; border:1px solid #22302A; border-radius:16px;
  background:#111a15; margin-bottom:14px; }
.brand { font-size:1.25rem; }
.brand span { color:#8FA396; font-size:.8rem; margin-left:10px; }
.pill { font-size:.8rem; color:#B8F26B; background:rgba(184,242,107,.1);
  padding:6px 12px; border-radius:99px; }
.card { background:#151F1A; border:1px solid #22302A; border-radius:16px;
  padding:16px 18px; margin-bottom:10px; }
.card .lbl { color:#8FA396; font-size:.75rem; text-transform:uppercase; letter-spacing:.8px; }
.card .val { font-size:1.9rem; font-weight:800; margin-top:4px; }
.card .sub { color:#8FA396; font-size:.82rem; }
.hero { border:1px solid #22302A; border-radius:20px; padding:30px 34px; margin:14px 0;
  background: linear-gradient(120deg, rgba(184,242,107,.12), rgba(94,234,212,.05)); }
.hero h1 { margin:0 0 6px 0; font-size:2.1rem; line-height:1.2; }
.hero p { margin:0; color:#8FA396; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- state default ----------------
st.session_state.setdefault('speed', 2)
st.session_state.setdefault('conf', 0.75)
st.session_state.setdefault('ref', pd.DataFrame({
    'dap': [10, 20, 30, 40, 50],
    'tinggi_cm': [4.0, 9.0, 15.0, 20.0, 22.0],
    'lebar_daun_cm': [2.5, 5.0, 8.0, 10.5, 11.5]}))   # GANTI dengan hasil ukur sendiri

# ---------------- helper ----------------
def log(aksi, detail=''):
    wib = dt.timezone(dt.timedelta(hours=7))
    row = pd.DataFrame([{'Waktu': dt.datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S'),
                         'Aktivitas': aksi, 'Detail': detail}])
    row.to_csv(HIST, mode='a', header=not os.path.exists(HIST), index=False)

def card(label, value, sub=''):
    st.markdown(f'<div class="card"><div class="lbl">{label}</div>'
                f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)

def theme(ch, h=240, title=''):
    return (ch.properties(height=h, title=alt.TitleParams(title, color='#E6F0E9',
                          anchor='start', fontSize=14))
              .configure(background='transparent')
              .configure_view(strokeWidth=0)
              .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID,
                              domainColor=GRID, tickColor=GRID)
              .configure_legend(labelColor='#E6F0E9', titleColor=MUTED, orient='bottom'))

def live_chart(df, col, color, title, dom):
    ch = alt.Chart(df).mark_area(line={'color': color, 'strokeWidth': 2.5},
                                 color=color, opacity=.18, interpolate='monotone').encode(
        x=alt.X('t:Q', axis=alt.Axis(labels=False, ticks=False, title=None)),
        y=alt.Y(f'{col}:Q', scale=alt.Scale(domain=dom), title=None))
    return theme(ch, 230, title)

def get_phase(m, thr):
    if m >= thr['Siap Panen']: return 'Siap Panen'
    if m >= thr['Mendekati Panen']: return 'Mendekati Panen'
    return 'Masa Pertumbuhan'

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

raw, daily = load_data()

# ---------------- header + navigasi ----------------
st.markdown('<div class="topbar"><div class="brand">🌱 <b>Pakcoy</b>Lab'
            '<span>growth intelligence</span></div>'
            '<div class="pill">● Sistem aktif · LSTM + CNN</div></div>',
            unsafe_allow_html=True)
page = st.segmented_control("Menu", PAGES, default=PAGES[0],
                            label_visibility="collapsed") or PAGES[0]
if st.session_state.get('last_page') != page:
    log('Membuka halaman', page)
    st.session_state['last_page'] = page

# ================= BERANDA =================
if page == 'Beranda':
    st.markdown('<div class="hero"><h1>Pantau tumbuh pakcoy,<br>tebak waktu panennya.</h1>'
                '<p>Sensor, prediksi LSTM, dan analisis gambar CNN dalam satu layar.</p></div>',
                unsafe_allow_html=True)
    k = st.columns(4)
    with k[0]: card('Hari tercatat', f"{len(daily)}", 'data harian')
    with k[1]: card('Kematangan akhir', f"{daily['maturity'].iloc[-1]:.0f}%", daily['phase'].iloc[-1])
    with k[2]: card('Rata-rata kelembaban', f"{raw['moisture'].mean():.0f}%", 'tanah')
    with k[3]: card('Rata-rata suhu', f"{raw['temp'].mean():.1f}°C", 'lingkungan')

    ch = alt.Chart(daily).mark_circle(size=45).encode(
        x=alt.X('dap:Q', title='DAP'),
        y=alt.Y('maturity:Q', title='Kematangan (%)', scale=alt.Scale(zero=False)),
        color=alt.Color('phase:N', legend=alt.Legend(title=None),
                        scale=alt.Scale(domain=list(PHASE_COLOR), range=list(PHASE_COLOR.values()))),
        tooltip=['dap', 'maturity', 'phase'])
    st.altair_chart(theme(ch, 280, 'Kurva kematangan per fase'), theme=None, width='stretch')

    st.markdown('#### Kalkulator pertumbuhan')
    ref = st.session_state['ref'].dropna().sort_values('dap')
    if len(ref) < 2:
        st.warning('Tabel acuan minimal 2 baris (atur di Pengaturan).')
    else:
        dap = st.slider('Usia tanaman (DAP)', 10, 50, 30)
        c1, c2 = st.columns(2)
        with c1: card('Tinggi tanaman', f"{np.interp(dap, ref['dap'], ref['tinggi_cm']):.1f} cm", 'estimasi dari tabel acuan')
        with c2: card('Lebar daun', f"{np.interp(dap, ref['dap'], ref['lebar_daun_cm']):.1f} cm", 'estimasi dari tabel acuan')

# ================= SENSOR LIVE =================
elif page == 'Sensor Live':
    st.markdown('### Sensor Live')
    st.caption('Simulasi pembacaan sensor dari dataset, diputar ulang otomatis.')
    st.toggle('Putar otomatis', value=True, key='play')

    def live_body():
        if 'idx' not in st.session_state: st.session_state.idx = 30
        if st.session_state.get('play', True):
            st.session_state.idx = (st.session_state.idx + 1) % len(raw)
        i = st.session_state.idx
        cur = raw.iloc[i]
        cond = str(cur['soil_cond'])
        ccol = {'Wet': MINT, 'Optimal': LIME, 'Dry': AMBER}.get(cond, MUTED)
        a, b, c, d = st.columns(4)
        with a: card('Kelembaban tanah', f"{cur['moisture']:.0f}%", f"DAP {int(cur['dap'])} · {str(cur['time'])[:5]}")
        with b: card('Suhu', f"{cur['temp']:.1f}°C", 'sensor lingkungan')
        with c: card('Kondisi tanah', cond, f'<span style="color:{ccol}">● status</span>')
        with d: card('Kematangan', f"{cur['maturity']:.0f}%", str(cur['phase']))
        win = raw.iloc[max(0, i - 35):i + 1].copy()
        win['t'] = range(len(win))
        g1, g2 = st.columns(2)
        g1.altair_chart(live_chart(win, 'moisture', MINT, 'Kelembaban tanah (%)', [40, 85]), theme=None, width='stretch')
        g2.altair_chart(live_chart(win, 'temp', AMBER, 'Suhu (°C)', [22, 33]), theme=None, width='stretch')

    st.fragment(run_every=float(st.session_state['speed']))(live_body)()

# ================= PREDIKSI =================
elif page == 'Prediksi':
    tab_lstm, tab_cnn = st.tabs(['📈 Prediksi LSTM · 4 hari', '🖼️ Analisis gambar CNN'])

    with tab_lstm:
        min_dap = int(daily['dap'].min()) + WINDOW - 1
        max_dap = int(daily['dap'].max())
        end_dap = st.slider('Prediksi dibuat pada DAP', min_dap, max_dap, max_dap)
        last10 = daily[daily['dap'] <= end_dap].tail(WINDOW)

        st.markdown('**Data sensor 10 hari terakhir (input model)**')
        st.dataframe(last10[['dap', 'moisture', 'temp', 'maturity', 'phase']].round(2),
                     hide_index=True, width='stretch')

        if st.button('Jalankan prediksi', type='primary'):
            model, scaler, thr = load_lstm()
            arr = scaler.transform(last10[FEATURES])
            out = model.predict(arr[None], verbose=0)[0]
            mn, mx = scaler.data_min_[3], scaler.data_max_[3]
            out = np.clip(out * (mx - mn) + mn, 0, 100)
            res = pd.DataFrame({'Hari': [f'+{i}' for i in range(1, 5)],
                                'DAP': [end_dap + i for i in range(1, 5)],
                                'Kematangan (%)': np.round(out, 1),
                                'Fase': [get_phase(v, thr) for v in out]})
            st.session_state['lstm'] = {'res': res, 'hist': last10}
            log('Prediksi LSTM', f'DAP {end_dap} → ' +
                ', '.join(f'DAP {d}: {v}%' for d, v in zip(res['DAP'], res['Kematangan (%)'])))

        if 'lstm' in st.session_state:
            res, hist = st.session_state['lstm']['res'], st.session_state['lstm']['hist']
            cols = st.columns(4)
            for col, (h, d, v, f_) in zip(cols, zip(res['Hari'], res['DAP'], res['Kematangan (%)'], res['Fase'])):
                with col:
                    pth = P(f"sample_images/{f_.replace(' ', '_')}.jpg")
                    if os.path.exists(pth): st.image(pth)
                    card(f'{h} hari · DAP {d}', f'{v:.1f}%', f_)

            plot = pd.concat([
                pd.DataFrame({'dap': hist['dap'], 'val': hist['maturity'], 'seri': 'Aktual'}),
                pd.DataFrame({'dap': res['DAP'], 'val': res['Kematangan (%)'], 'seri': 'Prediksi'})])
            ch = alt.Chart(plot).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X('dap:Q', title='DAP'),
                y=alt.Y('val:Q', title='Kematangan (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('seri:N', legend=alt.Legend(title=None),
                                scale=alt.Scale(domain=['Aktual', 'Prediksi'], range=[MUTED, LIME])))
            st.altair_chart(theme(ch, 260, 'Aktual vs prediksi'), theme=None, width='stretch')
            st.markdown('**Tabel hasil**')
            st.dataframe(res, hide_index=True, width='stretch')

    with tab_cnn:
        st.caption(f"Foto di bawah batas keyakinan {st.session_state['conf']:.2f} atau kelas non-pakcoy akan ditolak.")
        f = st.file_uploader('Unggah foto tanaman', type=['jpg', 'jpeg', 'png', 'webp'])
        if f:
            key = f'{f.name}-{f.size}'
            if st.session_state.get('last_upload') != key:
                log('Upload gambar CNN', f.name)
                st.session_state['last_upload'] = key
            img = Image.open(f).convert('RGB')
            left, right = st.columns(2)
            left.image(img)
            if right.button('Analisis gambar', type='primary'):
                model, names = load_cnn()
                x = np.array(img.resize((IMG, IMG), Image.BILINEAR), dtype='float32')[None]
                p = model.predict(x, verbose=0)[0]
                i = int(np.argmax(p))
                if names[i] == 'Bukan_Pakcoy' or p[i] < st.session_state['conf']:
                    with right: card('Hasil analisis', '⚠️ Bukan pakcoy', f'tidak dikenali · {p[i]*100:.1f}%')
                    hasil = 'Bukan pakcoy'
                else:
                    with right: card('Hasil analisis', names[i].replace('_', ' '), f'keyakinan {p[i]*100:.1f}%')
                    hasil = names[i]
                dfp = pd.DataFrame({'Kelas': [n.replace('_', ' ') for n in names], 'Probabilitas': p})
                ch = alt.Chart(dfp).mark_bar(color=LIME, cornerRadiusEnd=4).encode(
                    y=alt.Y('Kelas:N', sort='-x', title=None),
                    x=alt.X('Probabilitas:Q', scale=alt.Scale(domain=[0, 1]), title=None))
                right.altair_chart(theme(ch, 180, 'Probabilitas per kelas'), theme=None, width='stretch')
                log('Analisis CNN', f'{f.name} → {hasil} ({p[i]*100:.1f}%)')

# ================= DATASET =================
elif page == 'Dataset':
    st.markdown('### Dataset')
    t1, t2 = st.tabs(['Data sensor', 'Galeri gambar'])
    with t1:
        lo, hi = int(raw['dap'].min()), int(raw['dap'].max())
        rng = st.slider('Rentang DAP', lo, hi, (lo, min(hi, lo + 30)))
        sub = raw[(raw['dap'] >= rng[0]) & (raw['dap'] <= rng[1])]
        st.caption(f'{len(sub)} baris dari {len(raw)}')
        st.dataframe(sub, hide_index=True, width='stretch')
    with t2:
        cols = st.columns(3)
        for col, c in zip(cols, ['Masa_Pertumbuhan', 'Mendekati_Panen', 'Siap_Panen']):
            pth = P(f'sample_images/{c}.jpg')
            if os.path.exists(pth): col.image(pth, caption=c.replace('_', ' '))

# ================= HISTORI =================
elif page == 'Histori':
    st.markdown('### Histori aktivitas')
    if os.path.exists(HIST):
        h = pd.read_csv(HIST).iloc[::-1]
        a, b = st.columns(2)
        with a: card('Total aktivitas', f'{len(h)}', 'tercatat')
        with b: card('Prediksi & analisis', f"{h['Aktivitas'].str.contains('Prediksi|Analisis').sum()}", 'LSTM + CNN')
        jenis = st.multiselect('Filter aktivitas', sorted(h['Aktivitas'].unique()))
        if jenis: h = h[h['Aktivitas'].isin(jenis)]
        st.dataframe(h, hide_index=True, width='stretch')
        st.download_button('Unduh CSV', h.to_csv(index=False), 'histori.csv')
    else:
        st.info('Belum ada aktivitas.')

# ================= PENGATURAN =================
elif page == 'Pengaturan':
    st.markdown('### Pengaturan')
    c1, c2 = st.columns(2)
    with c1:
        st.slider('Kecepatan simulasi live (detik per data)', 1, 5, key='speed')
    with c2:
        st.slider('Batas keyakinan CNN', 0.50, 0.95, step=0.05, key='conf')
    st.markdown('**Tabel acuan tinggi & lebar daun** (ganti dengan hasil pengukuran sendiri)')
    st.session_state['ref'] = st.data_editor(st.session_state['ref'], num_rows='dynamic',
                                             hide_index=True, width='stretch')
    if st.button('Hapus semua histori'):
        if os.path.exists(HIST): os.remove(HIST)
        log('Menghapus histori')
        st.rerun()
