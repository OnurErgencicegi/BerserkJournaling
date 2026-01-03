# main.py
import time
import streamlit as st
from st_keyup import st_keyup

# Modüller
from core.config import DEFAULT_MOOD
from core.predictor import MoodPredictor
from core.director import director_engine
from ui.styles import apply_custom_css

# Sayfa Ayarı
st.set_page_config ( page_title = "Berserk Journaling" , page_icon = "🗡️" , layout = "wide" )
apply_custom_css ()

# --- HAFIZA (SESSION STATE) ---
# Uygulama hafızasını başlatıyoruz
defaults = {
    "last_switch": 0 ,
    "current_path": None ,
    "next_type": "gif" ,
    "used_files": {} ,
    "last_mood": None ,
    "active_music_mood": None ,
    "pending_mood": None ,
    # Chooser için gerekli hafıza alanı
    "visual_history": {
        'current_mood': None ,
        'used_gifs': [] ,
        'used_imgs': []
    }
}

for k , v in defaults.items ():
    if k not in st.session_state:
        st.session_state[k] = v


# --- MODEL YÜKLEME ---
@st.cache_resource
def get_predictor():
    return MoodPredictor ()  # Argüman göndermiyoruz, predictor.py kendi yolunu biliyor


try:
    predictor = get_predictor ()
except Exception as e:
    st.error ( f"Model Yüklenemedi: {e}" )
    st.stop ()

# --- ARAYÜZ (LAYOUT) ---
st.title ( "Berserk Journaling 🗡️" )

# Ekranı ikiye bölüyoruz: Sol taraf (Yazı + Stats), Sağ taraf (Boşluk/Görsel alanı)
col_input , col_space = st.columns ( [1 , 2] )

with col_input:
    # 1. YAZI ALANI
    # debounce=300: Yazmayı bıraktıktan 300ms sonra çalışır
    user_text = st_keyup ( " " , key = "active_journal" , debounce = 300 ,
                           placeholder = "Guts gibi anlat..." )

    detected_mood = None
    probs = None

    # Yazı varsa analiz yap
    if user_text and user_text.strip ():
        try:
            detected_mood , probs = predictor.predict ( user_text )

            # Başlık
            st.markdown ( f"### 🔥 Vibe: {detected_mood.upper ()}" )

            # 2. İSTATİSTİKLER (TEXTBOX ALTINA EKLENDİ)
            if probs:
                st.markdown ( "---" )
                st.caption ( "Duygu Analizi:" )

                c1 , c2 = st.columns ( 2 )

                # Puanına göre sırala
                items = sorted ( list ( probs.items () ) , key = lambda x: x[1] , reverse = True )
                mid = (len ( items ) + 1) // 2

                with c1:
                    for m , p in items[:mid]:
                        if p > 0.01:  # %1 altındakileri gösterme
                            st.write ( f"**{m.capitalize ()}**: %{int ( p * 100 )}" )
                            st.progress ( p )
                with c2:
                    for m , p in items[mid:]:
                        if p > 0.01:
                            st.write ( f"**{m.capitalize ()}**: %{int ( p * 100 )}" )
                            st.progress ( p )

        except Exception as e:
            st.warning ( f"Analiz hatası: {e}" )

# --- ATMOSFER MOTORU ---
# Yazı yoksa son bilinen mood veya varsayılan mood
final_mood = detected_mood if detected_mood else (st.session_state.last_mood or DEFAULT_MOOD)

# Director'ı çağır (Arka planı ve müziği yönetir)
director_engine ( final_mood )

# Son mood'u kaydet
if detected_mood:
    st.session_state.last_mood = detected_mood

# --- CANLI DÖNGÜ (Heartbeat) ---
# Burası çok önemli. Sayfanın sürekli yenilenmesini sağlar.
# Süreyi 1 saniyeye düşürdüm ki sistem daha sık kontrol etsin, takılma olmasın.
time.sleep ( 1.0 )
st.rerun ()