# core/director.py
import os
import time
import streamlit as st
import streamlit.components.v1 as components

# Config, Utils ve Chooser
from core.config import MEDIA_PATH , DEFAULT_MOOD , SWITCH_DURATION_GIF , SWITCH_DURATION_IMG , MOOD_ASSETS
from core.utils import set_background
from core.gif_img_chooser import get_next_visual


def director_engine(detected_mood):
    """
    Müzik: Anında değişir.
    Görsel: Süre dolunca (GIF <-> IMG) değişir.
    """

    # 1. MOOD HAZIRLIĞI
    if detected_mood is None: detected_mood = DEFAULT_MOOD
    current_mood_key = detected_mood.lower ()
    mood_folder_name = detected_mood.title ()

    # 2. MÜZİK (ANINDA TEPKİ)
    st.session_state.active_music_mood = current_mood_key
    assets = MOOD_ASSETS.get ( current_mood_key , MOOD_ASSETS["neutral"] )
    if assets and assets.get ( "music_embed" ):
        with st.sidebar:
            st.markdown ( "---" )
            st.caption ( f"🎵 Çalıyor: {current_mood_key.title ()}" )
            components.html ( assets["music_embed"] , height = 200 )

    # 3. DOSYALARI TARAMA
    folder = os.path.normpath ( os.path.join ( MEDIA_PATH , mood_folder_name ) )
    if not os.path.exists ( folder ):
        folder = os.path.normpath ( os.path.join ( MEDIA_PATH , current_mood_key ) )
    if not os.path.exists ( folder ):
        folder = os.path.normpath ( os.path.join ( MEDIA_PATH , DEFAULT_MOOD ) )
        if not os.path.exists ( folder ): return

    all_files = os.listdir ( folder )
    gifs = [f for f in all_files if f.lower ().endswith ( '.gif' )]
    imgs = [f for f in all_files if f.lower ().endswith ( ('.jpg' , '.jpeg' , '.png') )]
    if not gifs and not imgs: return

    # 4. ZAMANLAMA VE STATE
    now = time.time ()
    if 'last_switch' not in st.session_state: st.session_state.last_switch = 0
    if 'current_path' not in st.session_state: st.session_state.current_path = None
    if 'last_mood_check' not in st.session_state: st.session_state.last_mood_check = None
    if 'showing_type' not in st.session_state: st.session_state.showing_type = 'gif'

    # 5. DEĞİŞİM KARARI
    mood_changed = (st.session_state.last_mood_check != current_mood_key)

    # Bekleme süresini belirle
    duration = SWITCH_DURATION_GIF if st.session_state.showing_type == 'gif' else SWITCH_DURATION_IMG

    # Ne kadar zaman geçti?
    elapsed = now - st.session_state.last_switch
    time_is_up = elapsed >= duration

    # --- DEBUG: Sidebar'a Geri Sayım Koyalım ---
    # Bu sayede sistemin donup donmadığını anlarsın
    with st.sidebar:
        if not mood_changed:
            remaining = int ( duration - elapsed )
            if remaining > 0:
                st.caption ( f"⏳ Değişime: {remaining} sn ({st.session_state.showing_type.upper ()})" )
            else:
                st.caption ( "🚀 Değişiyor..." )

    # EYLEM ZAMANI
    if mood_changed or time_is_up or st.session_state.current_path is None:

        # Hedef türü belirle
        target_type = st.session_state.showing_type

        if mood_changed:
            target_type = 'gif'  # Mood değişince GIF ile başla
            st.session_state.last_switch = 0
        elif time_is_up:
            target_type = 'image' if target_type == 'gif' else 'gif'  # Türü değiştir

        # Seçimi Chooser'a Yaptır
        selected_file = get_next_visual ( current_mood_key , gifs , imgs , target_type )

        if selected_file:
            full_path = os.path.join ( folder , selected_file )
            st.session_state.current_path = full_path

            # ÖNEMLİ: Seçilen dosyanın gerçek uzantısına bakarak türü güncelle
            # Çünkü biz 'image' istemiş olabiliriz ama chooser 'gif' (fallback) dönmüş olabilir.
            ext = os.path.splitext ( selected_file )[1].lower ()
            real_type = 'gif' if ext == '.gif' else 'image'

            st.session_state.showing_type = real_type
            st.session_state.last_switch = now
            st.session_state.last_mood_check = current_mood_key

            set_background ( full_path )

    # Değişim zamanı değilse mevcudu koru
    elif st.session_state.current_path:
        set_background ( st.session_state.current_path )