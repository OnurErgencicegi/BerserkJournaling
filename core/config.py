import os

# ==========================================
# 1. UYGULAMA AYARLARI (Frontend & Logic)
# ==========================================

# Medya klasörünün yolu
MEDIA_PATH = r"C:/Users/onur_/PycharmProjects/BerserkJournaling/gifs"

# Varsayılan duygu
DEFAULT_MOOD = "Joy"

# Görsel değişim süreleri (saniye cinsinden)
SWITCH_DURATION_GIF = 20
SWITCH_DURATION_IMG = 10

# ==========================================
# 2. ATMOSFER AYARLARI (Müzik & Efektler)
# ==========================================
# Not: SFX ve Overlay şu an için 'None' yapıldı.
# İleride dosya eklediğinde buraya dosya adını (örn: "rain.mp3") yazabilirsin.

MOOD_ASSETS = {
    "sadness": {
        "music_embed": '''<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/2019232265&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>''' ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    } ,

    "joy": {
        "music_embed": '''<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1431682624&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>''' ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    } ,

    "love": {
        "music_embed": '''<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1064067817&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>''' ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    } ,

    "anger": {
        "music_embed": '''<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/483125619&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>''' ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    } ,

    "surprise": {
        "music_embed": '''<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1023202747&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>''' ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    } ,

    "fear": {
        "music_embed": '''<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/2028556952&color=%23ff5500&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>''' ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    } ,

    # Eğer model "neutral" döndürürse hata almamak için:
    "neutral": {
        "music_embed": None ,
        "sfx": None ,
        "overlay": None ,
        "overlay_opacity": 0
    }
}


# ==========================================
# 3. MODEL EĞİTİM AYARLARI (Backend / AI)
# ==========================================
class Config:
    """
    Model eğitimi ve parametreleri için kullanılan yapılandırma.
    """
    DATASET_URL = "parulpandey/emotion-dataset"
    MAX_LEN = 50
    BATCH_SIZE = 16
    EMBED_DIM = 100
    HIDDEN_DIM = 64
    LEARNING_RATE = 0.001
    EPOCHS = 15
    DROPOUT = 0.4

    LABEL_MAP = {
        'sadness': 0 , 'joy': 1 , 'love': 2 ,
        'anger': 3 , 'fear': 4 , 'surprise': 5
    }