# core/utils.py
import re
import os
import pandas as pd
import kagglehub
from core.config import Config


def clean_text(text):
    text = str ( text ).lower ()
    text = re.sub ( r'[^\w\s]' , '' , text )
    return text


def get_word2idx():
    """
    Kaggle'dan veriyi indirir ve kelime sözlüğünü (vocabulary) oluşturur.
    Predictor ve Model bu sözlüğe ihtiyaç duyar.
    """
    print ( "--- Veri Seti ve Sözlük Hazırlanıyor ---" )
    try:
        path = kagglehub.dataset_download ( Config.DATASET_URL )
    except Exception as e:
        print ( f"Veri indirme hatası: {e}" )
        return {"<PAD>": 0 , "<UNK>": 1}  # Hata olursa boş dönmesin

    # CSV dosyasını bul
    csv_files = [f for f in os.listdir ( path ) if f.endswith ( '.csv' )]
    if not csv_files:
        raise FileNotFoundError ( "İndirilen klasörde CSV dosyası bulunamadı." )

    df = pd.read_csv ( os.path.join ( path , csv_files[0] ) )

    # Sütun isimlerini düzelt (text, content, tweet vb.)
    df.columns = [c.lower () for c in df.columns]
    if 'text' not in df.columns:
        if 'content' in df.columns:
            df.rename ( columns = {'content': 'text'} , inplace = True )
        elif 'tweet' in df.columns:
            df.rename ( columns = {'tweet': 'text'} , inplace = True )

    # Kelime sözlüğü oluştur
    word2idx = {"<PAD>": 0 , "<UNK>": 1}
    idx = 2

    # Eğer 'text' sütunu bulunduysa işle, yoksa boş geç
    if 'text' in df.columns:
        for text in df['text']:
            cleaned = clean_text ( str ( text ) )
            for word in cleaned.split ():
                if word not in word2idx:
                    word2idx[word] = idx
                    idx += 1
    else:
        print ( "Uyarı: 'text' sütunu bulunamadı, sözlük sadece temel etiketlerle oluşturuldu." )

    return word2idx


def text_to_indices(text , word2idx , max_len):
    cleaned = clean_text ( text )
    tokens = cleaned.split ()
    indices = []
    for word in tokens:
        indices.append ( word2idx.get ( word , word2idx["<UNK>"] ) )

    if len ( indices ) > max_len:
        indices = indices[:max_len]
    else:
        indices += [word2idx["<PAD>"]] * (max_len - len ( indices ))
    return indices