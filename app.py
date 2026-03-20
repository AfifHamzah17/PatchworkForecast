# 1. SUPPRESS WARNINGS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from hijridate import Gregorian
from fpdf import FPDF
import re

# --- 2. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Forecasting UMKM Patchwork", layout="wide")

# --- 3. HELPER FUNCTIONS ---

def format_rupiah(value):
    return f"Rp {int(value):,}".replace(",", ".")

def get_hijri_events(g_date):
    try:
        h_date = Gregorian(g_date.year, g_date.month, g_date.day).to_hijri()
        is_ramadhan = (h_date.month == 9)
        is_idul_fitri = (h_date.month == 10 and h_date.day <= 3)
        is_idul_adha = (h_date.month == 12 and h_date.day == 10)
        return is_ramadhan, is_idul_fitri, is_idul_adha
    except:
        return False, False, False

def check_hot_season(target_date):
    month = target_date.month
    day = target_date.day
    is_ramadhan, is_idul_fitri, is_idul_adha = get_hijri_events(target_date)
    is_natal = (month == 12 and day >= 20)
    is_ajaran_baru = (month == 7) or (month == 1)
    
    if is_ramadhan: return True, "Ramadhan"
    elif is_idul_fitri: return True, "Idul Fitri"
    elif is_idul_adha: return True, "Idul Adha"
    elif is_natal: return True, "Natal"
    elif is_ajaran_baru: return True, "Ajaran Baru"
    else: return False, "Biasa"

# Fungsi untuk membersihkan Emoji agar bisa masuk PDF (Latin-1)
def clean_text_for_pdf(text):
    # Hapus karakter non-ASCII (termasuk emoji) atau ganti dengan placeholder
    return re.sub(r'[^\x00-\xff]', '', text)

# Fungsi PDF Generator
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Laporan Prediksi AI UMKM Patchwork', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

def create_pdf_report(results, date_str):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, f"Tanggal Prediksi: {date_str}", ln=True)
    pdf.ln(5)
    
    for res in results:
        # Bersihkan teks dari Emoji sebelum ke PDF
        prod_name = clean_text_for_pdf(res['Produk'])
        status_txt = clean_text_for_pdf(res['Status'])
        rekomendasi_txt = clean_text_for_pdf(res['Rekomendasi'])
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"Produk: {prod_name} | Status: {status_txt}", ln=True)
        
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, f"  - Skenario Pesimis (P10): {res['P10']} Unit", ln=True)
        pdf.cell(0, 8, f"  - Skenario Realistis (P50): {res['P50']} Unit", ln=True)
        pdf.cell(0, 8, f"  - Skenario Optimis (P90): {res['P90']} Unit", ln=True)
        pdf.cell(0, 8, f"  - Gap Kapasitas: {res['Gap']} Unit", ln=True)
        pdf.cell(0, 8, f"  - Est. Biaya Lembur: Rp {res['Biaya_Lembur']:,}", ln=True)
        
        pdf.set_font("Arial", 'I', 9)
        pdf.multi_cell(0, 5, f"  - Rekomendasi: {rekomendasi_txt}")
        pdf.ln(5)
        
    return pdf.output(dest='S').encode('latin-1')

# --- 4. LOAD ASSETS ---
@st.cache_resource
def load_model_assets():
    try:
        model = load_model('model_lstm.h5')
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        return None, None, None

@st.cache_data
def load_business_params():
    try:
        df = pd.read_csv('dataset_synthetic_umkm.csv')
        if 'Produk' not in df.columns:
            prod_cols = [c for c in df.columns if c.startswith('Prod_')]
            def get_prod_name(row):
                for c in prod_cols:
                    if row[c] == 1 or row[c] == True: return c.replace('Prod_', '')
                return 'Unknown'
            df['Produk'] = df.apply(get_prod_name, axis=1)

        params_df = df.groupby('Produk').agg({
            'Waktu_Produksi': lambda x: x.mode()[0],
            'Biaya_Lembur': lambda x: x.mode()[0],
            'Lead_Time': lambda x: x.mode()[0],
            'Ongkir_Bahan_Baku': lambda x: x.mode()[0],
            'Max_Lembur_Bulan': lambda x: x.mode()[0],
            'Max_TK_Tambahan': lambda x: x.mode()[0]
        })
        return params_df.to_dict('index'), list(params_df.index)
    except:
        dummy = {
            'Totebag': {'Waktu_Produksi': 24, 'Biaya_Lembur': 50000, 'Lead_Time': 2, 'Ongkir_Bahan_Baku': 20000, 'Max_Lembur_Bulan': 20, 'Max_TK_Tambahan': 0}
        }
        return dummy, list(dummy.keys())

model, scaler_X, scaler_y = load_model_assets()
PRODUCT_PARAMS, produk_list = load_business_params()

# --- 5. SIDEBAR INPUT ---
st.sidebar.title("⚙️ Parameter Prediksi")

# Tombol Hari Ini
if st.sidebar.button("📅 Set ke Hari Ini", type="secondary"):
    st.session_state.selected_date = datetime.now()
    st.rerun()

# Date Input dengan State
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = datetime.now()

selected_date = st.sidebar.date_input("Pilih Tanggal Prediksi:", st.session_state.selected_date)
selected_date_dt = pd.to_datetime(selected_date)

# Auto-detect Season
is_hot_season, season_name = check_hot_season(selected_date_dt)
if is_hot_season:
    st.sidebar.success(f"🌡️ Terdeteksi Musim: {season_name}")
    default_season_check = True
else:
    default_season_check = False

is_peak = st.sidebar.checkbox("Tandai sebagai Musim Puncak", value=default_season_check)
is_event = st.sidebar.checkbox("Ada Event Khusus (Bazaar)", value=False)

# Default Produk Kosong
selected_products = st.sidebar.multiselect("Pilih Produk:", produk_list, default=[])

st.sidebar.divider()
st.sidebar.subheader("📊 Data Historis")
lag_demand = st.sidebar.number_input("Penjualan 7 Hari Lalu", min_value=0, value=5)
rolling_demand = st.sidebar.number_input("Rata-rata Penjualan 7 Hari", min_value=0.0, value=5.0)

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1: predict_btn = st.button("🔮 Prediksi")
with col_btn2: reset_btn = st.button("🔄 Reset")

# --- 6. LOGIKA UTAMA ---
if 'results' not in st.session_state:
    st.session_state.results = []

if reset_btn:
    st.session_state.results = []
    st.rerun()

if predict_btn and model is not None:
    if not selected_products:
        st.warning("⚠️ Pilih minimal 1 produk terlebih dahulu!")
    else:
        progress_bar = st.progress(0)
        temp_results = []
        
        for i, product in enumerate(selected_products):
            progress_bar.progress((i + 1) / len(selected_products))
            
            params = PRODUCT_PARAMS.get(product, {
                'Waktu_Produksi': 10, 'Biaya_Lembur': 60000, 'Lead_Time': 7,
                'Ongkir_Bahan_Baku': 20000, 'Max_Lembur_Bulan': 20, 'Max_TK_Tambahan': 1
            })
            
            prod_cols = {f'Prod_{p}': 1 if p == product else 0 for p in produk_list}
            input_data = {
                'Is_Peak_Season': int(is_peak),
                'Is_Event': int(is_event),
                'Waktu_Produksi': params['Waktu_Produksi'],
                'Biaya_Lembur': params['Biaya_Lembur'],
                'Lead_Time': params['Lead_Time'],
                'Ongkir_Bahan_Baku': params['Ongkir_Bahan_Baku'],
                'Max_Lembur_Bulan': params['Max_Lembur_Bulan'],
                'Max_TK_Tambahan': params['Max_TK_Tambahan'],
                'Demand_Lag_7': float(lag_demand),
                'Demand_Rolling_Mean_7': float(rolling_demand)
            }
            input_data.update(prod_cols)
            df_input = pd.DataFrame([input_data])
            
            try:
                df_input = df_input[scaler_X.feature_names_in_]
            except: pass
            
            scaled_input = scaler_X.transform(df_input)
            n_timesteps = 7
            X_predict = np.tile(scaled_input, (n_timesteps, 1)).reshape(1, n_timesteps, len(df_input.columns))
            pred_scaled = model.predict(X_predict, verbose=0)
            prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
            prediction = int(max(0, prediction))
            
            # --- LOGIKA TSSP & BUSINESS FIX ---
            std_dev = 3.79
            
            p50 = prediction
            p10 = int(prediction - (1.28 * std_dev))
            if p10 < 1: p10 = 1
            p90 = int(prediction + (1.28 * std_dev))
            
            kapasitas_normal = 10 
            gap = p50 - kapasitas_normal
            
            if p50 > kapasitas_normal:
                jam_lembur = abs(gap) * 1.5
                total_biaya = int(jam_lembur * params['Biaya_Lembur'])
                
                if jam_lembur > params['Max_Lembur_Bulan']:
                    status = "⚠️ OVERLOAD KRITIS"
                    rekomendasi = (
                        f"**Analisis:** Demand ({p50} unit) jauh melampaui kapasitas.\n\n"
                        f"**Mengapa?** Gap produksi mencapai **{gap} unit**. Kapasitas lembur normal ({params['Max_Lembur_Bulan']} jam) tidak cukup.\n\n"
                        f"**Solusi:** Sistem menyarankan mencari **Tenaga Borongan** atau **Menolak Order** sebanyak {abs(int(gap - (params['Max_Lembur_Bulan']/1.5)))} unit."
                    )
                else:
                    status = "🚧 BUTUH LEMBUR"
                    rekomendasi = (
                        f"**Analisis:** Target ({p50} unit) di atas kapasitas normal ({kapasitas_normal} unit).\n\n"
                        f"**Mengapa?** Terdapat gap kebutuhan sebesar **{gap} unit**.\n\n"
                        f"**Solusi:** Jalankan overtime selama **{jam_lembur:.1f} jam** untuk mengejar target."
                    )
            else:
                gap = 0
                jam_lembur = 0
                total_biaya = 0
                status = "✅ KAPASITAS AMAN"
                if p50 <= 2:
                    rekomendasi = (
                        f"**Analisis:** Demand diprediksi sangat rendah ({p50} unit).\n\n"
                        f"**Mengapa?** Kemungkinan bukan musim puncak atau histori penjualan rendah.\n\n"
                        f"**Solusi:** Hindari produksi berlebih (overstock). Cukup siapkan 'Ready Stock' minimal."
                    )
                else:
                    rekomendasi = (
                        f"**Analisis:** Target ({p50} unit) di bawah atau sama dengan kapasitas normal.\n\n"
                        f"**Mengapa?** Kapasitas kerja reguler mencukupi tanpa perlu ekspansi.\n\n"
                        f"**Solusi:** Fokus pada efisiensi biaya dan jadwal reguler."
                    )
            
            temp_results.append({
                'Produk': product,
                'P10': p10, 'P50': p50, 'P90': p90,
                'Gap': gap,
                'Status': status,
                'Rekomendasi': rekomendasi,
                'Biaya_Lembur': total_biaya,
                'Ongkir': params['Ongkir_Bahan_Baku']
            })
        
        st.session_state.results = temp_results

# --- 7. TAMPILKAN HASIL (UI + CHART) ---
st.title("📊 Dashboard Forecasting & Optimization")

if st.session_state.results:
    st.divider()
    
    for res in st.session_state.results:
        with st.container():
            col_info, col_chart = st.columns([2, 1])
            
            with col_info:
                st.subheader(f"📦 {res['Produk']}")
                st.caption(f"Status Sistem: {res['Status']}")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("📉 Pesimis (P10)", f"{res['P10']} Unit", help="Safety Stock")
                c2.metric("🎯 Realistis (P50)", f"{res['P50']} Unit", delta=f"Gap: {res['Gap']}", delta_color="inverse" if res['Gap'] > 0 else "normal")
                c3.metric("📈 Optimis (P90)", f"{res['P90']} Unit", help="Max Capacity Planning")
                
                st.markdown(res['Rekomendasi'])
                
                with st.expander("💰 Rincian Biaya"):
                    col_cost1, col_cost2 = st.columns(2)
                    col_cost1.write(f"**Estimasi Biaya Lembur:** {format_rupiah(res['Biaya_Lembur'])}")
                    col_cost2.write(f"**Ongkir Bahan Baku:** {format_rupiah(res['Ongkir'])}")
            
            with col_chart:
                # --- FIX: Added unique key for charts ---
                
                # 1. Bar Chart
                chart_df = pd.DataFrame({
                    'Skenario': ['P10 (Min)', 'P50 (Target)', 'P90 (Max)'],
                    'Jumlah Unit': [res['P10'], res['P50'], res['P90']]
                })
                fig_bar = px.bar(chart_df, x='Skenario', y='Jumlah Unit', 
                             color='Skenario', 
                             color_discrete_map={'P10 (Min)': 'blue', 'P50 (Target)': 'green', 'P90 (Max)': 'red'},
                             text_auto=True)
                fig_bar.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{res['Produk']}")

                # 2. Gauge Chart (Utilisasi Kapasitas)
                util_percent = (res['P50'] / 10) * 100
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = util_percent,
                    title = {'text': "Utilisasi Kapasitas (%)"},
                    gauge = {
                        'axis': {'range': [None, 150]},
                        'steps': [
                            {'range': [0, 80], 'color': "lightgray"},
                            {'range': [80, 100], 'color': "gray"},
                            {'range': [100, 150], 'color': "red"}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 100} }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{res['Produk']}")

            st.divider()

    # --- DOWNLOAD PDF ---
    st.subheader("📥 Unduh Laporan")
    pdf_bytes = create_pdf_report(st.session_state.results, selected_date.strftime("%Y-%m-%d"))
    
    st.download_button(
        label="⬇️ Download Laporan PDF",
        data=pdf_bytes,
        file_name=f"Laporan_Prediksi_{selected_date.strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

else:
    st.info("👈 Silakan pilih produk di sidebar lalu klik **Prediksi**.")