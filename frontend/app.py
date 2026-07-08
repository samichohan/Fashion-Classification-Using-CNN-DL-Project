import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="wide"
)

# ---------------- BACKEND URL ---------------- #
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    BACKEND_URL = "http://localhost:8000"

# ---------------- CUSTOM DARK THEME CSS ---------------- #
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0E14;
        color: #E8E9ED;
    }

    h1, h2, h3 {
        color: #F5F5F7;
        font-family: 'Segoe UI', sans-serif;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00D9C0;
        margin-bottom: 0;
    }

    .subtitle {
        color: #8B8FA3;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.8rem;
    }

    .card {
        background-color: #131720;
        border: 1px solid #232838;
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1rem;
    }

    .prediction-title {
        font-size: 2rem;
        font-weight: 800;
        color: #00D9C0;
        margin-bottom: 0.2rem;
    }

    .confidence-text {
        color: #8B8FA3;
        font-size: 0.95rem;
    }

    .top3-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #232838;
    }

    .top3-label {
        color: #E8E9ED;
        font-weight: 600;
    }

    .top3-conf {
        color: #00D9C0;
        font-weight: 700;
    }

    .stButton > button {
        background-color: #00D9C0;
        color: #0B0E14;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #00B8A3;
        color: #0B0E14;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown('<p class="main-title">👗 Fashion Item Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a clothing image or paste a URL to identify the fashion category using a CNN model.</p>', unsafe_allow_html=True)

# ---------------- BACKEND HEALTH CHECK ---------------- #
def check_backend_health():
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return res.status_code == 200 and res.json().get("model_loaded", False)
    except Exception:
        return False

backend_healthy = check_backend_health()
if not backend_healthy:
    st.warning("⚠️ Backend API is not reachable right now. Predictions may fail. Please try again shortly.")

# ---------------- LAYOUT ---------------- #
left, right = st.columns([1, 1])

image_to_predict = None
source_label = None
uploaded_file = None
image_url = None

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📁 Upload Image", "🔗 Image URL"])

    with tab1:
        uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_predict = Image.open(uploaded_file)
            source_label = "upload"
            st.image(image_to_predict, caption="Uploaded Image", width=250)

    with tab2:
        image_url = st.text_input("Paste an image URL", placeholder="https://example.com/shirt.jpg")
        if image_url:
            try:
                preview_response = requests.get(image_url, timeout=10)
                image_to_predict = Image.open(io.BytesIO(preview_response.content))
                source_label = "url"
                st.image(image_to_predict, caption="Image from URL", width=250)
            except Exception:
                st.error("Could not load an image from that URL. Please check the link.")

    predict = st.button("🔍 Classify Item", disabled=(image_to_predict is None))
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if predict:
        with st.spinner("Analyzing image..."):
            try:
                if source_label == "upload":
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/predict/upload", files=files, timeout=30)
                else:
                    response = requests.post(f"{BACKEND_URL}/predict/url", json={"url": image_url}, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    predicted_label = result["predicted_label"]
                    confidence = result["confidence"]
                    top3 = result["top3"]

                    st.markdown(f'<p class="prediction-title">{predicted_label}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="confidence-text">Confidence: {confidence*100:.1f}%</p>', unsafe_allow_html=True)
                    st.progress(confidence)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Top 3 Predictions**")
                    for item in top3:
                        st.markdown(
                            f'''
                            <div class="top3-row">
                                <span class="top3-label">{item["label"]}</span>
                                <span class="top3-conf">{item["confidence"]*100:.1f}%</span>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                else:
                    error_detail = response.json().get("detail", "Something went wrong.")
                    st.error(f"❌ {error_detail}")

            except requests.exceptions.RequestException:
                st.error("❌ Could not reach the backend API. Please try again later.")
    else:
        st.info("Upload an image or paste a URL, then click **Classify Item** to see the result.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built with FastAPI + TensorFlow + Streamlit · Fashion MNIST Classification System")
