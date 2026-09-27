import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Laptop Price Predictor | AI Machine Learning App",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-right: 4px solid #3B82F6;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "best_laptop_price_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_pipeline()

# GPU mapping
gpu_models_map = {
    "NVIDIA": [
        "GeForce RTX 4060", "GeForce RTX 4070", "GeForce RTX 4080",
        "GeForce RTX 4090", "GeForce RTX 3050", "GeForce MX450",
        "GeForce MX550", "RTX A2000", "RTX A4000", "RTX A5000"
    ],
    "AMD": ["Radeon Vega", "Radeon 680M/780M"],
    "Intel": ["Iris Xe Graphics", "UHD Graphics"],
    "Apple": ["Apple M-Series Integrated"]
}

st.markdown("<div class='main-title'>💻 Laptop Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Machine Learning model (LightGBM) trained on comprehensive hardware specs to accurately estimate market prices.</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Architecture & Info")
    st.markdown("""
    - **Algorithm:** LightGBM Regressor
    - **Determination Coefficient ($R^2$):** ~0.90+
    - **Feature Engineering:** PPI (Pixels Per Inch), aggregated storage (GB), GPU classification, and memory types.
    - **Currencies Supported:** EUR (€), USD ($), SAR (SAR).
    """)
    st.markdown("---")
    st.markdown("💡 *Tip: Tweak hardware specs such as RAM, GPU tier, or Screen Resolution to observe real-time price impact.*")

# Main Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏢 Brand & Form Factor")
    company = st.selectbox(
        "Brand / Manufacturer",
        ["Apple", "Asus", "Dell", "HP", "Lenovo", "MSI", "Razer", "Acer"],
        index=0
    )
    
    typename = st.selectbox(
        "Form Factor / Type",
        ["Ultrabook", "Gaming", "Notebook", "Workstation", "2 in 1 Convertible"],
        index=0
    )
    
    opsys = st.selectbox(
        "Operating System",
        ["macOS" if company == "Apple" else "Windows 11", "Windows 10", "Linux", "No OS", "macOS"],
        index=0
    )
    
    st.subheader("⚡ Processor & RAM (CPU & Memory)")
    default_cpu_idx = 2 if company == "Apple" else 1
    cpu_brand = st.selectbox("CPU Brand", ["AMD", "Intel", "Apple"], index=default_cpu_idx)
    cpu_speed = st.slider("CPU Clock Speed (GHz)", min_value=1.1, max_value=4.0, value=3.2 if company == "Apple" else 2.6, step=0.1)
    ram_gb = st.select_slider("RAM Capacity (GB)", options=[4, 8, 16, 24, 32, 64], value=16)

with col2:
    st.subheader("🖥️ Display & Dimensions")
    inches = st.slider("Screen Size (Inches)", min_value=13.0, max_value=17.3, value=14.0, step=0.1)
    resolution = st.selectbox(
        "Screen Resolution",
        ["1920x1080 (FHD)", "2560x1440 (2K / QHD)", "2560x1600 (Retina / QHD+)", "3840x2160 (4K UHD)", "1366x768 (HD)"],
        index=1
    )
    clean_res = resolution.split()[0]
    
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        touchscreen = st.checkbox("Touchscreen", value=False)
    with col_feat2:
        ips_panel = st.checkbox("IPS Panel", value=True)
        
    weight_kg = st.number_input("Weight (kg)", min_value=0.8, max_value=4.5, value=1.4, step=0.1)

    st.subheader("🎮 Graphics & Storage (GPU & Drives)")
    default_gpu_b = "Apple" if company == "Apple" else "NVIDIA"
    gpu_brand = st.selectbox("GPU Brand", ["NVIDIA", "AMD", "Intel", "Apple"], index=["NVIDIA", "AMD", "Intel", "Apple"].index(default_gpu_b))
    gpu_model_options = gpu_models_map.get(gpu_brand, ["UHD Graphics"])
    gpu_model = st.selectbox("GPU Model", gpu_model_options, index=0)
    
    col_st1, col_st2 = st.columns(2)
    with col_st1:
        primary_type = st.selectbox("Primary Storage Type", ["SSD", "HDD", "eMMC"], index=0)
        primary_gb = st.selectbox("Primary Storage Capacity (GB)", [128, 256, 512, 1024, 2048], index=2)
    with col_st2:
        sec_has = st.checkbox("Add Secondary Drive", value=False)
        if sec_has:
            sec_type = "HDD"
            sec_gb = st.selectbox("Secondary Capacity (GB)", [500, 1024, 2048], index=1)
        else:
            sec_type = "None"
            sec_gb = 0

st.markdown("---")

# Prediction Execution
if st.button("🚀 Predict Laptop Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'Company': company,
        'TypeName': typename,
        'Inches': float(inches),
        'ScreenResolution': clean_res,
        'Cpu_Brand': cpu_brand,
        'Cpu_Speed_GHz': float(cpu_speed),
        'Ram_GB': int(ram_gb),
        'PrimaryStorage_GB': int(primary_gb),
        'PrimaryStorage_Type': primary_type,
        'SecondaryStorage_GB': int(sec_gb),
        'SecondaryStorage_Type': sec_type,
        'Gpu_Brand': gpu_brand,
        'Gpu_Model': gpu_model,
        'OpSys': opsys,
        'Weight_kg': float(weight_kg),
        'Touchscreen': 1 if touchscreen else 0,
        'IPS_Panel': 1 if ips_panel else 0
    }])
    
    # Feature Engineering
    res_split = input_df['ScreenResolution'].str.extract(r'(\d+)x(\d+)')
    input_df['X_res'] = res_split[0].astype(float)
    input_df['Y_res'] = res_split[1].astype(float)
    input_df['PPI'] = np.round(np.sqrt(input_df['X_res']**2 + input_df['Y_res']**2) / input_df['Inches'], 2)
    input_df['TotalStorage_GB'] = input_df['PrimaryStorage_GB'] + input_df['SecondaryStorage_GB']
    input_df['Has_SSD'] = (input_df['PrimaryStorage_Type'] == 'SSD').astype(int)
    input_df['Is_Dedicated_GPU'] = input_df['Gpu_Brand'].isin(['NVIDIA']).astype(int)
    
    clean_features = input_df.drop(columns=['ScreenResolution', 'X_res', 'Y_res'])
    
    pred_log = model.predict(clean_features)
    pred_eur = float(np.expm1(pred_log)[0])
    pred_usd = pred_eur * 1.08
    pred_sar = pred_eur * 4.05
    
    st.markdown("### 🎯 Estimated Market Value")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric(label="Estimated Price (EUR)", value=f"€ {pred_eur:,.2f}")
    with r2:
        st.metric(label="Estimated Price (USD)", value=f"$ {pred_usd:,.2f}")
    with r3:
        st.metric(label="Estimated Price (SAR)", value=f"{pred_sar:,.2f} SAR")
        
    st.markdown(f"""
    <div class='info-box'>
        <b>💡 Computed Specifications & Derived Features:</b><br>
        Pixel Density: <b>{input_df['PPI'].iloc[0]:.1f} PPI</b> | 
        Total Storage: <b>{input_df['TotalStorage_GB'].iloc[0]} GB</b> | 
        GPU Tier: <b>{'Dedicated GPU (dGPU)' if input_df['Is_Dedicated_GPU'].iloc[0] == 1 else 'Integrated GPU (iGPU)'}</b>
    </div>
    """, unsafe_allow_html=True)
