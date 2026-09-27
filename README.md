---
title: Laptop Price Predictor
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# 💻 Laptop Price Predictor

An interactive end-to-end Machine Learning web application powered by **LightGBM** to accurately predict laptop market prices based on modern hardware specifications.

## 🚀 Key Highlights & Architecture
- **Trained Model**: High-performance LightGBM Regressor pipeline achieving an **$R^2$ score over 0.90**.
- **Automated Feature Engineering**:
  - Screen resolution decomposition & **Pixels Per Inch (PPI)** calculation.
  - Multi-tier storage aggregation (**TotalStorage_GB**, SSD identification flag).
  - GPU classification (**Dedicated dGPU vs Integrated iGPU**).
- **Multi-Currency Output**: Instant real-time conversions into **EUR (€)**, **USD ($)**, and **SAR**.
- **Interactive UI**: Built with Streamlit with responsive modern controls and dynamic GPU models mapping.

## 🛠️ Project Structure
```text
├── app.py                         # Streamlit interactive application
├── best_laptop_price_model.joblib # Trained LightGBM pipeline artifact
├── requirements.txt               # Application dependencies
├── Dockerfile                     # Docker configuration for cloud deployment
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## ⚙️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/laptop-price-predictor.git
   cd laptop-price-predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

## 🐳 Docker Deployment
To run with Docker:
```bash
docker build -t laptop-price-predictor .
docker run -p 8501:8501 laptop-price-predictor
```

## 📄 License
This project is licensed under the [MIT License](LICENSE).

