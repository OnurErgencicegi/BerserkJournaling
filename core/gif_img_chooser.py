# core/gif_img_chooser.py
import random
import streamlit as st


def get_next_visual(mood , gifs , imgs , target_type):
    """
    Belirtilen türde (GIF/IMG) sıradaki görseli seçer.
    Daha önce gösterilenleri hatırlar ve liste bitene kadar tekrar etmez.
    """

    # --- 1. HAFIZA (STATE) BAŞLATMA ---
    # Eğer hafızada 'visual_history' yoksa oluştur
    if 'visual_history' not in st.session_state:
        st.session_state.visual_history = {
            'current_mood': None ,
            'used_gifs': [] ,
            'used_imgs': []
        }

    history = st.session_state.visual_history

    # --- 2. MOOD DEĞİŞTİ Mİ? ---
    # Eğer mood değiştiyse hafızayı sıfırla, yeni sayfa aç
    if history['current_mood'] != mood:
        history['current_mood'] = mood
        history['used_gifs'] = []
        history['used_imgs'] = []

    selected_file = None

    # --- 3. SEÇİM MANTIĞI (NO-REPEAT) ---

    # --- DURUM A: Hedef GIF ise ---
    if target_type == 'gif' and gifs:
        # Kullanılmamış olanları bul (Mevcutlar - Kullanılanlar)
        unused = [g for g in gifs if g not in history['used_gifs']]

        # Eğer hepsi kullanıldıysa listeyi sıfırla (Başa dön)
        if not unused:
            history['used_gifs'] = []
            unused = gifs

        selected_file = random.choice ( unused )
        history['used_gifs'].append ( selected_file )

    # --- DURUM B: Hedef RESİM ise ---
    elif target_type == 'image' and imgs:
        unused = [i for i in imgs if i not in history['used_imgs']]

        if not unused:
            history['used_imgs'] = []
            unused = imgs

        selected_file = random.choice ( unused )
        history['used_imgs'].append ( selected_file )

    # --- DURUM C: YEDEK PLAN (Fallback) ---
    # Eğer resim sırası geldi ama klasörde resim yoksa, mecburen GIF göster (veya tam tersi)
    if selected_file is None:
        if gifs:
            selected_file = random.choice ( gifs )  # Rastgele al geç
        elif imgs:
            selected_file = random.choice ( imgs )

    return selected_file