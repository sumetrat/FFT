import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 🎨 WEB APP SETUP
# ==========================================
st.set_page_config(page_title="Vernier Pro Web Analyzer", layout="wide")

st.title("📊 Vernier FFT Analyzer (Web Edition)")
st.markdown("---")

# ==========================================
# 🛠️ CORE LOGIC (Same Logic as Desktop App)
# ==========================================
def calculate_fft_high_precision(times, values):
    n = len(values)
    if n == 0: return np.array([]), np.array([]), 0
    dt = np.mean(np.diff(times))
    fs = 1.0 / dt
    
    # Detrend & Windowing
    x = np.arange(n)
    fit = np.polyfit(x, values, 1)
    trend = np.polyval(fit, x)
    values_detrended = values - trend
    window = np.hanning(n)
    values_windowed = values_detrended * window
    
    # Ultra Padding for Smoothness
    n_padded = 65536 
    fft_vals = np.abs(np.fft.fft(values_windowed, n=n_padded))[:n_padded//2]
    freqs = np.fft.fftfreq(n_padded, dt)[:n_padded//2]
    fft_vals = fft_vals * (4.0 / n)
    
    return freqs, fft_vals, fs

# ==========================================
# 📂 FILE UPLOAD SECTION
# ==========================================
uploaded_file = st.file_uploader("📂 Upload Vernier CSV File Here", type=["csv"])

if uploaded_file is not None:
    try:
        # อ่านไฟล์ผ่าน Pandas (เพราะง่ายกว่าใน Web)
        df = pd.read_csv(uploaded_file, header=None) # อ่านแบบไม่มี header ไปก่อน
        
        # พยายามหาคอลัมน์ที่เป็นตัวเลข
        # สมมติว่า Vernier CSV ทั่วไป คอลัมน์ 0 = เวลา, คอลัมน์ 1 = ค่า
        # ข้ามบรรทัดที่เป็น Text
        df_clean = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_clean) > 0:
            times = df_clean.iloc[:, 0].values
            values = df_clean.iloc[:, 1].values
            
            # คำนวณ
            freqs, fft_vals, fs = calculate_fft_high_precision(times, values)
            
            # Peak Finding
            idx = np.argmax(fft_vals)
            peak_freq = freqs[idx]
            
            # --- DISPLAY RESULT ---
            st.success(f"✅ Data Loaded: {len(values)} samples | Sampling Rate: {fs:.0f} Hz")
            
            # โชว์ค่า Peak ตัวใหญ่ๆ
            st.metric(label="🎯 Peak Frequency", value=f"{peak_freq:.2f} Hz")
            
            # สร้างกราฟ
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            plt.subplots_adjust(hspace=0.4)
            
            # Waveform
            ax1.plot(times, values, color='#00A7E1')
            ax1.set_title("Waveform", fontweight='bold')
            ax1.grid(True, alpha=0.5)
            
            # FFT
            ax2.plot(freqs, fft_vals, color='#9E45FF')
            ax2.fill_between(freqs, fft_vals, color='#9E45FF', alpha=0.3)
            ax2.set_title("FFT Spectrum (Ultra-High Precision)", fontweight='bold')
            ax2.set_xlabel("Frequency (Hz)")
            ax2.grid(True, alpha=0.5)
            
            # Auto Zoom Logic
            threshold = np.max(fft_vals) * 0.1
            sig_idx = np.where(fft_vals > threshold)[0]
            view_limit = 2000
            if len(sig_idx) > 0:
                view_limit = max(2000, freqs[sig_idx[-1]] + 500)
            ax2.set_xlim(0, min(view_limit, fs/2))
            ax2.set_ylim(bottom=0)

            # ส่งกราฟไปแสดงบนเว็บ
            st.pyplot(fig)
            
        else:
            st.error("❌ Error: Could not parse numerical data from CSV.")
            
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("👋 Please upload a CSV file to begin analysis.")