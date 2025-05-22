import streamlit as st
import os
import sys
from pathlib import Path
import logging
import base64
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Fix for Streamlit event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Disable Streamlit's file watcher completely
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Set page config
st.set_page_config(
    page_title="Accent Analysis Tool",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add Google Fonts
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background
set_background('assets/dark-web-pictures-vhvyvp7xgv1pnnef.jpg')

# Modern glassmorphism and centering
st.markdown("""
    <style>
    .centered-card {
        max-width: 900px;
        margin: 6vh auto 0 auto;
        background: rgba(30, 30, 30, 0.60);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 3rem 2.5rem 2.5rem 2.5rem;
    }
    .stColumns {
        gap: 2rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
    }
    p, label, .stMarkdown, .stTextInput label, .stTextInput>div>div>input {
        color: #fff !important;
        font-family: 'Inter', sans-serif;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        background: rgba(255,255,255,0.85);
        color: #222;
        font-size: 1.1rem;
        padding: 0.75rem;
        border: none;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #43cea2 0%, #185a9d 100%);
        color: #fff;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: background 0.2s, transform 0.2s;
        margin-bottom: 2rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #185a9d 0%, #43cea2 100%);
        transform: translateY(-2px) scale(1.03);
    }
    .stTextArea>div>div>textarea {
        background: rgba(255,255,255,0.85);
        color: #222;
        border-radius: 10px;
        font-size: 1.1rem;
        min-height: 250px !important;
        height: 300px !important;
        width: 100% !important;
        resize: vertical;
    }
    .stSuccess, .stInfo, .stError, .stWarning {
        border-radius: 10px;
        font-size: 1.1rem;
        color: #fff !important;
    }
    .stMetric {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        color: #fff !important;
    }
    .stMarkdown {
        color: #fff !important;
    }
    .stMarkdown a {
        color: #43cea2 !important;
    }
    .stMarkdown a:hover {
        color: #185a9d !important;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_analyzer():
    """Initialize the accent analyzer with proper error handling."""
    try:
        from accent_analyzer import AccentAnalyzer
        return AccentAnalyzer()
    except Exception as e:
        logger.error(f"Error initializing analyzer: {str(e)}")
        st.error(f"Failed to initialize the accent analyzer: {str(e)}")
        return None

def main():
    with st.container():
        st.markdown('<div class="centered-card">', unsafe_allow_html=True)
        st.title("🎙️ Accent Analysis Tool")
        st.write("Upload a video URL to analyze the speaker's accent.")

        # Initialize analyzer in session state if not exists
        if 'analyzer' not in st.session_state:
            with st.spinner("Initializing accent analyzer..."):
                st.session_state.analyzer = initialize_analyzer()

        # Input for video URL
        url = st.text_input(
            "Enter video URL (Google Drive or direct MP4 link):",
            label_visibility="visible",
            help="Supported formats: Google Drive or direct MP4 links"
        )

        if st.button("Analyze Accent", type="primary"):
            if not st.session_state.analyzer:
                st.error("Accent analyzer not initialized. Please refresh the page.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            if not url:
                st.warning("Please enter a video URL.")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            with st.spinner("Analyzing accent..."):
                try:
                    results = st.session_state.analyzer.analyze_accent(url)
                    if 'error' in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        st.success(f"Accent: {results['accent']}")
                        st.metric("Confidence", f"{results['confidence']:.1%}")
                        st.info("Transcription")
                        st.text_area("", results['transcription'], height=300)
                        st.write(results['explanation'])
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"An error occurred during analysis: {str(e)}")
        st.markdown("---")
        st.markdown("Built for REM Waste Accent Analysis Challenge")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 