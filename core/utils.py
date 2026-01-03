# core/utils.py
import re
import os
import base64
import pickle
import streamlit as st

# Yeni modüler yapıdan importlar
from ui.styles import inject_background_css

# --- YOL AYARLARI ---
# Projenin ana dizinini buluyoruz (core klasörünün bir üstü)
BASE_DIR = os.path.dirname ( os.path.dirname ( os.path.abspath ( __file__ ) ) )
VOCAB_PATH = os.path.join ( BASE_DIR , "models" , "vocab.pkl" )


# ==========================================
# 1. DOSYA VE ARAYÜZ İŞLEMLERİ
# ==========================================

def get_base64(path):
    """
    Verilen dosya yolundaki resmi okur ve Base64 formatına çevirir.
    """
    try:
        with open ( path , "rb" ) as f:
            return base64.b64encode ( f.read () ).decode ()
    except Exception:
        return ""


def set_background(image_path):
    """
    Görsel yolunu alır, okur, mime tipini belirler ve
    UI katmanına (styles.py) CSS olarak gönderir.
    """
    if not image_path or not os.path.exists ( image_path ):
        return

    # 1. Base64 verisini al
    b64_image = get_base64 ( image_path )
    if not b64_image:
        return

    # 2. Uzantıya göre Mime Type belirle
    ext = os.path.splitext ( image_path )[1].lower ()
    mime_type = "image/gif" if ext == ".gif" else "image/jpeg"

    # 3. UI katmanındaki fonksiyonu çağırarak CSS'i bas
    inject_background_css ( b64_image , mime_type )


# ==========================================
# 2. YAPAY ZEKA VE METİN İŞLEMLERİ
# ==========================================

def clean_text(text):
    text = str ( text ).lower ()
    text = re.sub ( r'[^\w\s]' , '' , text )
    return text


def get_word2idx():
    """
    Eğitim sırasında (train.py) oluşturulan 'vocab.pkl' dosyasını yükler.
    Artık internetten indirme yapmaz.
    """
    if os.path.exists ( VOCAB_PATH ):
        try:
            with open ( VOCAB_PATH , 'rb' ) as f:
                return pickle.load ( f )
        except Exception as e:
            print ( f"Sözlük yükleme hatası: {e}" )
            return {"<PAD>": 0 , "<UNK>": 1}
    else:
        # Eğer dosya yoksa (henüz train.py çalıştırılmadıysa) hata vermesin diye:
        print ( f"UYARI: {VOCAB_PATH} bulunamadı! Lütfen önce train.py dosyasını çalıştırın." )
        return {"<PAD>": 0 , "<UNK>": 1}


def text_to_indices(text , word2idx , max_len):
    """
    Metni temizler ve sözlükteki karşılıklarına (sayılara) çevirir.
    """
    cleaned = clean_text ( text )
    tokens = cleaned.split ()
    indices = []

    for word in tokens:
        # Kelime sözlükte yoksa <UNK> (1) kullan
        indices.append ( word2idx.get ( word , word2idx.get ( "<UNK>" , 1 ) ) )

    # Uzunluk sabitleme (Padding / Truncating)
    if len ( indices ) > max_len:
        indices = indices[:max_len]
    else:
        indices += [word2idx.get ( "<PAD>" , 0 )] * (max_len - len ( indices ))

    return indices