

import streamlit as st
import pandas as pd
from datetime import datetime
from core.predictor import MoodPredictor

# Sayfa Ayarları
st.set_page_config (
    page_title = "Berserk Journaling" ,
    page_icon = "🧠" ,
    layout = "wide" ,
    initial_sidebar_state = "expanded"
)

# --- CSS İLE ÖZELLEŞTİRME (OPSİYONEL) ---
st.markdown ( """
<style>
    .stTextArea textarea {font-size: 16px !important;}
    .mood-card {padding: 15px; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 10px;}
</style>
""" , unsafe_allow_html = True )

# --- OTURUM (SESSION) YÖNETİMİ ---
# Geçmiş kayıtları hafızada tutmak için
if 'history' not in st.session_state:
    st.session_state['history'] = []


# Modeli bir kez yükle (Cache mekanizması)
@st.cache_resource
def load_predictor():
    return MoodPredictor ()


predictor = load_predictor ()

# ==========================================
# SOL MENÜ (SIDEBAR) - AYARLAR VE GEÇMİŞ
# ==========================================
with st.sidebar:
    st.header ( "⚙️ Ayarlar" )

    # Toggle Butonlar
    show_details = st.toggle ( "Detaylı Analizi Göster" , value = True )
    dark_mode_analysis = st.toggle ( "Karanlık Mod Analizi" , value = False )

    st.divider ()  # Çizgi çek

    st.header ( "📚 Geçmiş Günlükler" )

    # Geçmiş listesini göster
    if len ( st.session_state['history'] ) > 0:
        for i , entry in enumerate ( reversed ( st.session_state['history'] ) ):
            # Her bir geçmiş kaydı için bir buton/expander
            with st.expander ( f"{entry['date']} - {entry['mood'].upper ()}" ):
                st.write ( entry['text'][:50] + "..." )  # Metnin başını göster
                st.caption ( f"Skor: %{entry['score'] * 100:.1f}" )
    else:
        st.info ( "Henüz bir giriş yapılmadı." )

# ==========================================
# ANA SAYFA (MAIN AREA)
# ==========================================
st.title ( "Berserk Journaling 🗡️" )
st.subheader ( "Bugün nasıl hissediyorsun?" )

# Kullanıcıdan Metin Alma
user_text = st.text_area ( "İçini dök..." , height = 150 , placeholder = "Bugün proje yüzünden biraz gergindim ama..." )

col1 , col2 = st.columns ( [1 , 5] )
with col1:
    analyze_btn = st.button ( "Analiz Et" , use_container_width = True , type = "primary" )

if analyze_btn and user_text:
    with st.spinner ( 'Duygular analiz ediliyor...' ):
        # Modelden tahmin al
        dominant_mood , probabilities = predictor.predict ( user_text )

        if dominant_mood:
            # Sonucu Geçmişe Ekle
            score = probabilities[dominant_mood]
            new_entry = {
                'date': datetime.now ().strftime ( "%H:%M" ) ,
                'text': user_text ,
                'mood': dominant_mood ,
                'score': score
            }
            st.session_state['history'].append ( new_entry )

            # --- SONUÇ EKRANI ---
            st.success ( f"Baskın Duygu: **{dominant_mood.upper ()}**" )

            # Eğer ayarlardan 'Detaylı Analizi Göster' açıksa
            if show_details:
                st.write ( "---" )
                st.write ( "#### Duygu Dağılımı" )

                # Grafikleri 2 kolon halinde gösterelim
                cols = st.columns ( len ( probabilities ) )
                for idx , (mood , prob) in enumerate ( probabilities.items () ):
                    with cols[idx]:
                        st.metric ( label = mood.capitalize () , value = f"%{prob * 100:.1f}" )
                        # İlerleme çubuğu
                        st.progress ( prob )
        else:
            st.error ( "Bir hata oluştu. Lütfen tekrar deneyin." )