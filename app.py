import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. ⚙️ SETUP (ต้องอยู่บรรทัดแรกสุดของ Streamlit)
# ==========================================
st.set_page_config(
    page_title="Vernier Pro Web Analyzer", 
    layout="wide",  # <--- พระเอกของเราอยู่นี่ครับ!
    page_icon="📊"
)

# ==========================================
# 🎨 HEADER & STYLES
# ==========================================
st.title("📊 Vernier FFT Analyzer (Wide Dashboard Edition)")
st.markdown("---")

# ==========================================
# 🛠️ CORE LOGIC
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
    
    # Ultra Padding
    n_padded = 65536 
    fft_vals = np.abs(np.fft.fft(values_windowed, n=n_padded))[:n_padded//2]
    freqs = np.fft.fftfreq(n_padded, dt)[:n_padded//2]
    fft_vals = fft_vals * (4.0 / n)
    
    return freqs, fft_vals, fs

# ==========================================
# 📂 SIDEBAR (ย้ายปุ่มโหลดไปไว้ด้านซ้าย)
# ==========================================
with st.sidebar:
    st.header("📂 Control Panel")
    uploaded_file = st.file_uploader("Upload Vernier CSV", type=["csv"])
    st.info("💡 Tip: Use 'Wide Mode' in settings for better view.")

# ==========================================
# 📊 MAIN DASHBOARD
# ==========================================
if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file, header=None)
        df_clean = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if len(df_clean) > 0:
            times = df_clean.iloc[:, 0].values
            values = df_clean.iloc[:, 1].values
            
            # Calculate
            freqs, fft_vals, fs = calculate_fft_high_precision(times, values)
            
            # Peak Finding
            idx = np.argmax(fft_vals)
            peak_freq = freqs[idx]
            
            # --- 1. SHOW STATS (แถบข้อมูลด้านบน) ---
            # แบ่งเป็น 3 คอลัมน์สำหรับโชว์ตัวเลข
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("🎯 Peak Frequency", f"{peak_freq:.2f} Hz", delta="High Precision")
            col_stat2.metric("📦 Data Points", f"{len(values)} samples")
            col_stat3.metric("⏱️ Sampling Rate", f"{fs:.0f} Hz")
            
            st.markdown("---")

            # --- 2. SHOW GRAPHS (จัดกราฟซ้าย-ขวา) ---
            # แบ่งหน้าจอเป็น 2 ฝั่งเท่ากัน
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.subheader("🌊 Waveform (Time Domain)")
                fig1, ax1 = plt.subplots(figsize=(6, 4)) # ปรับขนาดให้พอดี column
                ax1.plot(times, values, color='#00A7E1')
                ax1.grid(True, alpha=0.5)
                ax1.set_xlim(times[0], times[-1])
                st.pyplot(fig1)
                
            with col_graph2:
                st.subheader("📊 FFT Spectrum (Frequency Domain)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(freqs, fft_vals, color='#9E45FF')
                ax2.fill_between(freqs, fft_vals, color='#9E45FF', alpha=0.3)
                ax2.grid(True, alpha=0.5)
                
                # Auto Zoom Logic
                threshold = np.max(fft_vals) * 0.1
                sig_idx = np.where(fft_vals > threshold)[0]
                view_limit = 2000
                if len(sig_idx) > 0:
                    view_limit = max(2000, freqs[sig_idx[-1]] + 500)
                ax2.set_xlim(0, min(view_limit, fs/2))
                ax2.set_ylim(bottom=0)
                
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.markdown("### 👋 Welcome to Vernier Pro Analyzer")
    st.write("Please upload a CSV file from the sidebar to start analysis.")