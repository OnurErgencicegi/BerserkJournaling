# train.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset , TensorDataset
from collections import Counter
import pickle
import re

# Kendi modüllerimiz
from core.config import Config
from core.model import EmotionCNN

# --- YOL AYARLARI ---
BASE_DIR = os.path.dirname ( os.path.abspath ( __file__ ) )
DATA_PATH = os.path.join ( BASE_DIR , "dataset.csv" )  # Ana dizindeki csv
MODELS_DIR = os.path.join ( BASE_DIR , "models" )  # models klasörü
MODEL_SAVE_PATH = os.path.join ( MODELS_DIR , "emotion_model.pth" )
VOCAB_SAVE_PATH = os.path.join ( MODELS_DIR , "vocab.pkl" )


# --- YARDIMCI FONKSİYONLAR ---
def clean_text(text):
    text = str ( text ).lower ()
    text = re.sub ( r'[^a-z\s]' , '' , text )
    return text


def build_vocab(texts):
    """Veri setindeki kelimelerden sözlük oluşturur."""
    word2idx = {"<PAD>": 0 , "<UNK>": 1}
    idx = 2
    for text in texts:
        for word in text.split ():
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    return word2idx


def text_to_indices_local(text , word2idx , max_len):
    """Metni sayılara çevirir (Yerel fonksiyon)."""
    tokens = text.split ()
    indices = [word2idx.get ( token , word2idx["<UNK>"] ) for token in tokens]
    if len ( indices ) < max_len:
        indices += [word2idx["<PAD>"]] * (max_len - len ( indices ))
    else:
        indices = indices[:max_len]
    return indices


class EmotionDataset ( Dataset ):
    def __init__(self , X , y):
        self.X = torch.tensor ( X , dtype = torch.long )
        self.y = torch.tensor ( y , dtype = torch.long )

    def __len__(self):
        return len ( self.X )

    def __getitem__(self , idx):
        return self.X[idx] , self.y[idx]


def train_pipeline():
    print ( "--- 1. Hazırlık ve Veri Okuma ---" )
    device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
    print ( f"Cihaz: {device}" )

    # Klasör kontrolü
    if not os.path.exists ( MODELS_DIR ):
        os.makedirs ( MODELS_DIR )

    # CSV Okuma
    if not os.path.exists ( DATA_PATH ):
        print ( f"HATA: {DATA_PATH} bulunamadı!" )
        return

    df = pd.read_csv ( DATA_PATH )

    # Sütun isimlerini düzeltme
    df.columns = [c.lower () for c in df.columns]
    if 'text' not in df.columns:
        if 'content' in df.columns:
            df.rename ( columns = {'content': 'text'} , inplace = True )
        elif 'tweet' in df.columns:
            df.rename ( columns = {'tweet': 'text'} , inplace = True )

    if 'label' not in df.columns:
        if 'sentiment' in df.columns:
            df.rename ( columns = {'sentiment': 'label'} , inplace = True )
        elif 'emotion' in df.columns:
            df.rename ( columns = {'emotion': 'label'} , inplace = True )

    # Temizlik
    print ( "Metinler temizleniyor..." )
    df['text'] = df['text'].apply ( clean_text )

    # Label Mapping
    if df['label'].dtype == 'O':  # Object (String) ise
        df = df[df['label'].isin ( Config.LABEL_MAP.keys () )]
        df['label_id'] = df['label'].map ( Config.LABEL_MAP )
    else:
        df['label_id'] = df['label']

    # Boş verileri at
    df = df.dropna ( subset = ['text' , 'label_id'] )

    # --- SÖZLÜK OLUŞTURMA VE KAYDETME ---
    print ( "Kelime sözlüğü oluşturuluyor..." )
    word2idx = build_vocab ( df['text'].values )
    print ( f"Kelime Sayısı: {len ( word2idx )}" )

    with open ( VOCAB_SAVE_PATH , 'wb' ) as f:
        pickle.dump ( word2idx , f )
    print ( f"Sözlük kaydedildi: {VOCAB_SAVE_PATH}" )

    # Veriyi Hazırla
    X = [text_to_indices_local ( t , word2idx , Config.MAX_LEN ) for t in df['text']]
    y = df['label_id'].values

    # Sınıf Ağırlıkları (Dengesiz veri için)
    class_counts = Counter ( y )
    total_samples = len ( y )
    num_classes = len ( Config.LABEL_MAP )
    class_weights = []
    for i in range ( num_classes ):
        count = class_counts.get ( i , 0 )
        weight = total_samples / (num_classes * count) if count > 0 else 1.0
        class_weights.append ( weight )

    weights_tensor = torch.FloatTensor ( class_weights ).to ( device )

    # Dataset ve Loader
    dataset = EmotionDataset ( X , y )
    train_loader = DataLoader ( dataset , batch_size = Config.BATCH_SIZE , shuffle = True )

    print ( "\n--- 2. Model Başlatılıyor ---" )
    # Model init (dropout parametresi eklendi)
    model = EmotionCNN (
        vocab_size = len ( word2idx ) + 1 ,
        embed_dim = Config.EMBED_DIM ,
        hidden_dim = Config.HIDDEN_DIM ,
        output_dim = len ( Config.LABEL_MAP ) ,
        dropout = Config.DROPOUT
    )
    model.to ( device )

    # Loss Fonksiyonu
    criterion = nn.CrossEntropyLoss ( weight = weights_tensor , label_smoothing = 0.1 )
    optimizer = optim.Adam ( model.parameters () , lr = Config.LEARNING_RATE )

    print ( "\n--- 3. Eğitim Başlıyor ---" )
    model.train ()
    for epoch in range ( Config.EPOCHS ):
        total_loss = 0
        for batch_X , batch_y in train_loader:
            batch_X , batch_y = batch_X.to ( device ) , batch_y.to ( device )

            optimizer.zero_grad ()
            predictions = model ( batch_X )
            loss = criterion ( predictions , batch_y )
            loss.backward ()
            optimizer.step ()

            total_loss += loss.item ()

        print ( f"Epoch {epoch + 1}/{Config.EPOCHS} | Loss: {total_loss / len ( train_loader ):.4f}" )

    # Kaydet
    torch.save ( model.state_dict () , MODEL_SAVE_PATH )
    print ( f"\n--- Model Başarıyla Kaydedildi: {MODEL_SAVE_PATH} ---" )


if __name__ == "__main__":
    train_pipeline ()