# ==========================================================
# FILE: app.py
# APLIKASI AI FORECASTING UMKM PATCHWORK (DYNAMIC CSV VERSION)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Forecasting UMKM Patchwork", layout="wide")

# --- 2. LOAD ASSETS (Model, Scaler, CSV) ---

# Fungsi Load Model
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

# Fungsi Load Parameter Bisnis dari CSV
@st.cache_data
def load_business_params():
    try:
        df = pd.read_csv('dataset_synthetic_umkm.csv')
        
        # Deteksi nama kolom produk (Bisa 'Produk' atau One-Hot 'Prod_...')
        # Jika kolom 'Produk' tidak ada, kita rekonstruksi dari One-Hot columns
        if 'Produk' not in df.columns:
            prod_cols = [c for c in df.columns if c.startswith('Prod_')]
            def get_prod_name(row):
                for c in prod_cols:
                    if row[c] == 1 or row[c] == True:
                        return c.replace('Prod_', '')
                return 'Unknown'
            df['Produk'] = df.apply(get_prod_name, axis=1)

        # Agregat parameter unik per produk (ambil modus)
        params_df = df.groupby('Produk').agg({
            'Waktu_Produksi': lambda x: x.mode()[0],
            'Biaya_Lembur': lambda x: x.mode()[0],
            'Lead_Time': lambda x: x.mode()[0],
            'Ongkir_Bahan_Baku': lambda x: x.mode()[0],
            'Max_Lembur_Bulan': lambda x: x.mode()[0],
            'Max_TK_Tambahan': lambda x: x.mode()[0]
        })
        
        # Konversi ke Dictionary Dictionary
        params_dict = params_df.to_dict('index')
        # Ambil list nama produk
        produk_list = list(params_dict.keys())
        
        return params_dict, produk_list

    except Exception as e:
        st.warning(f"⚠️ File dataset_synthetic_umkm.csv tidak ditemukan atau error. Menggunakan data dummy. Error: {e}")
        # Fallback Data Dummy jika CSV hilang
        dummy_data = {
            'Totebag': {'Waktu_Produksi': 24, 'Biaya_Lembur': 50000, 'Lead_Time': 2, 'Ongkir_Bahan_Baku': 20000, 'Max_Lembur_Bulan': 20, 'Max_TK_Tambahan': 0},
            'Sajadah': {'Waktu_Produksi': 24, 'Biaya_Lembur': 70000, 'Lead_Time': 10, 'Ongkir_Bahan_Baku': 10000, 'Max_Lembur_Bulan': 30, 'Max_TK_Tambahan': 2}
        }
        return dummy_data, list(dummy_data.keys())

model, scaler_X, scaler_y = load_model_assets()
PRODUCT_PARAMS, produk_list = load_business_params()

# Helper Functions
def format_rupiah(value):
    return f"Rp {int(value):,}".replace(",", ".")

def check_hot_season(target_date):
    event_calendar = {
        'Ramadhan': [('2022-04-03', '2022-05-01'), ('2023-03-23', '2023-04-21'), ('2024-03-11', '2024-04-09'), ('2025-03-01', '2025-03-30')],
        'Natal':    [('2022-12-20', '2022-12-31'), ('2023-12-20', '2023-12-31'), ('2024-12-20', '2024-12-31')]
    }
    for event, ranges in event_calendar.items():
        for start, end in ranges:
            if pd.to_datetime(start) <= target_date <= pd.to_datetime(end):
                return True, event
    return False, "Biasa"

# --- 3. SIDEBAR INPUT ---
st.sidebar.title("⚙️ Parameter Prediksi")

selected_date = st.sidebar.date_input("Pilih Tanggal Prediksi:", datetime.now())
selected_date_dt = pd.to_datetime(selected_date)

is_hot_season, season_name = check_hot_season(selected_date_dt)
if is_hot_season:
    st.sidebar.success(f"🌡️ Terdeteksi Musim: {season_name}")
    default_season_check = True
else:
    default_season_check = False

is_peak = st.sidebar.checkbox("Tandai sebagai Musim Puncak", value=default_season_check)
is_event = st.sidebar.checkbox("Ada Event Khusus (Bazaar)", value=False)

# Multi-Select Produk
selected_products = st.sidebar.multiselect("Pilih Produk (Bisa >1):", produk_list, default=produk_list[0])

st.sidebar.divider()
st.sidebar.subheader("📊 Data Historis Pendukung")
lag_demand = st.sidebar.number_input("Penjualan 7 Hari Lalu", min_value=0, value=5)
rolling_demand = st.sidebar.number_input("Rata-rata Penjualan 7 Hari", min_value=0.0, value=5.0)

# Tombol Aksi
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    predict_btn = st.button("🔮 Prediksi")
with col_btn2:
    reset_btn = st.button("🔄 Reset")

# --- 4. LOGIKA UTAMA ---
if 'results' not in st.session_state:
    st.session_state.results = []

if reset_btn:
    st.session_state.results = []
    st.rerun()

if predict_btn and model is not None:
    if not selected_products:
        st.warning("Pilih minimal 1 produk!")
    else:
        progress_bar = st.progress(0)
        temp_results = []
        
        for i, product in enumerate(selected_products):
            progress_bar.progress((i + 1) / len(selected_products))
            
            # Ambil parameter dari Dictionary hasil baca CSV
            params = PRODUCT_PARAMS.get(product)
            if not params:
                continue # Skip jika produk tidak ditemukan di DB
            
            # Susun Input Data
            # 1. Fitur Biasa
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
            
            # 2. One-Hot Encoding Produk
            for p in produk_list:
                input_data[f'Prod_{p}'] = 1 if p == product else 0

            df_input = pd.DataFrame([input_data])
            
            # 3. Fix Feature Order (PENTING!)
            try:
                # Paksa urutan kolom sama persis dengan saat training
                df_input = df_input[scaler_X.feature_names_in_]
            except AttributeError:
                # Fallback untuk scaler versi lama
                pass
            
            # 4. Scaling & Reshape
            scaled_input = scaler_X.transform(df_input)
            n_timesteps = 7
            # Replikasi input untuk 7 timesteps (asumsi kondisi hari ini mewakili 7 hari terakhir)
            X_predict = np.tile(scaled_input, (n_timesteps, 1)).reshape(1, n_timesteps, len(df_input.columns))
            
            # 5. Predict
            pred_scaled = model.predict(X_predict, verbose=0)
            prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
            prediction = int(max(0, prediction))
            
            # --- LOGIKA TSSP ENHANCED ---
            std_dev = 3.79 # RMSE dari evaluasi model
            p10 = max(0, int(prediction - (1.28 * std_dev)))
            p50 = prediction
            p90 = int(prediction + (1.28 * std_dev))
            
            kapasitas_normal = 10 # Asumsi kapasitas normal per hari
            
            if p50 > kapasitas_normal:
                gap = p50 - kapasitas_normal
                jam_lembur = gap * 1.5 # Asumsi 1 jam lembur menghasilkan 1.5 unit
                biaya_lembur_total = int(jam_lembur * params['Biaya_Lembur'])
                
                if jam_lembur > params['Max_Lembur_Bulan']:
                    status = "⚠️ OVERLOAD"
                    tk_needed = math.ceil((p50 - kapasitas_normal) / 10) # Asumsi 1 org tambahan = 10 unit
                    rekomendasi = (f"Peringatan! Demand {p50} unit melebihi batas lembur. "
                                   f"Disarankan menambah **{tk_needed} tenaga kerja borongan**.")
                else:
                    status = "🚧 BUTUH LEMBUR"
                    rekomendasi = (f"Target produksi {p50} unit melebihi kapasitas normal. "
                                   f"Lakukan lembur **{jam_lembur:.0f} jam**.")
            else:
                gap = 0
                jam_lembur = 0
                biaya_lembur_total = 0
                status = "✅ AMAN"
                rekomendasi = "Kapasitas normal mencukupi. Fokus pada efisiensi biaya."
            
            temp_results.append({
                'Produk': product,
                'P10': p10, 'P50': p50, 'P90': p90,
                'Status': status,
                'Rekomendasi': rekomendasi,
                'Biaya_Lembur': biaya_lembur_total,
                'Ongkir': params['Ongkir_Bahan_Baku']
            })
        
        st.session_state.results = temp_results

# --- 5. TAMPILKAN HASIL (ENHANCED UI) ---
st.title("📊 AI Forecasting & Optimization System")

if st.session_state.results:
    st.divider()
    
    for res in st.session_state.results:
        with st.container():
            col_header, col_metric = st.columns([1, 3])
            
            with col_header:
                st.subheader(f"📦 {res['Produk']}")
                st.caption(f"Status: {res['Status']}")
            
            with col_metric:
                c1, c2, c3 = st.columns(3)
                c1.metric("📉 Pesimis (P10)", f"{res['P10']} Unit", help="Skenario terburuk (90% confidence lower bound).")
                c2.metric("🎯 Realistis (P50)", f"{res['P50']} Unit", delta="Target Utama", delta_color="off")
                c3.metric("📈 Optimis (P90)", f"{res['P90']} Unit", help="Skenario terbaik (90% confidence upper bound).")
            
            st.info(f"**🛠️ Rekomendasi TSSP:** {res['Rekomendasi']}")
            
            with st.expander("💰 Lihat Rincian Biaya Estimasi"):
                col_cost1, col_cost2 = st.columns(2)
                col_cost1.write(f"**Biaya Lembur Potensial:** {format_rupiah(res['Biaya_Lembur'])}")
                col_cost2.write(f"**Estimasi Ongkir Bahan:** {format_rupiah(res['Ongkir'])}")
                
            st.divider()

    # Tombol Download
    df_out = pd.DataFrame(st.session_state.results)
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Laporan CSV", csv, "laporan_prediksi.csv", "text/csv")

else:
    st.info("👈 Silakan atur parameter di sidebar lalu klik **Prediksi**.")