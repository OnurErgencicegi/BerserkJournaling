# ui/styles.py
import streamlit as st

def apply_custom_css():
    """Tüm sayfa stillerini yükler."""
    st.markdown("""
    <style>
        /* --- ARKA PLAN --- */
        [data-testid="stAppViewContainer"] {
            background-color: black;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* --- INPUT KUTUSU TASARIMI (KEYUP HACK) --- */
        div[data-testid="stKeyup"] {
            height: 60vh !important;
            overflow-y: auto !important;
        }

        div[data-testid="stKeyup"] input {
            height: 100% !important;
            min-height: 400px !important;
            width: 100% !important;
            white-space: pre-wrap !important;
            overflow-wrap: break-word !important;
            text-overflow: clip !important;
            overflow-y: auto !important;
            background-color: rgba(0, 0, 0, 0.6) !important;
            color: #E0E0E0 !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            font-size: 18px !important;
            line-height: 1.5 !important;
            display: block !important;
            align-content: flex-start !important;
        }

        div[data-testid="stKeyup"] input:focus {
            border-color: #FF4B4B !important;
            box-shadow: 0 0 15px rgba(255, 75, 75, 0.5);
        }

        div[data-testid="stKeyup"] label { display: none; }

        /* --- ANALİZ BARLARI --- */
        .mood-stat {
            font-size: 13px !important;
            color: rgba(255, 255, 255, 0.9) !important;
            margin-bottom: 2px !important;
            font-weight: bold;
        }
        .stProgress > div > div { height: 6px !important; }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: rgba(10, 10, 10, 0.95) !important;
            border-right: 1px solid #333;
        }
        h1 { text-shadow: 0 3px 5px black; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

def inject_background_css(b64_image, mime_type):
    """Hesaplanan görseli CSS olarak sayfaya basar."""
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:{mime_type};base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{ visibility: hidden; }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)