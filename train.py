# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader , Dataset
import kagglehub
import pandas as pd
import os
import numpy as np
from collections import Counter

from core.config import Config
from core.utils import get_word2idx , text_to_indices


class EmotionDataset ( Dataset ):
    def __init__(self , X , y):
        self.X = X
        self.y = y

    def __len__(self): return len ( self.X )

    def __getitem__(self , idx): return self.X[idx] , self.y[idx]


def train_pipeline():
    """
        Executes the full training pipeline:
        1. Downloads and preprocesses the dataset.
        2. Calculates class weights to handle imbalance.
        3. Trains the custom CNN model.
        4. Saves the trained model weights to disk.
        """
    print ( "--- 1. Hazırlık ve Ağırlık Hesaplama ---" )
    device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
    print ( f"Cihaz: {device}" )

    word2idx = get_word2idx ()
    path = kagglehub.dataset_download ( Config.DATASET_URL )
    csv_files = [f for f in os.listdir ( path ) if f.endswith ( '.csv' )]
    df = pd.read_csv ( os.path.join ( path , csv_files[0] ) )

    df.columns = [c.lower () for c in df.columns]
    if 'text' not in df.columns:
        if 'content' in df.columns:
            df.rename ( columns = {'content': 'text'} , inplace = True )
        elif 'tweet' in df.columns:
            df.rename ( columns = {'tweet': 'text'} , inplace = True )

    if df['label'].dtype == 'O':
        df = df[df['label'].isin ( Config.LABEL_MAP.keys () )]
        df['label_id'] = df['label'].map ( Config.LABEL_MAP )
    else:
        df['label_id'] = df['label']

    X = [text_to_indices ( t , word2idx , Config.MAX_LEN ) for t in df['text']]
    y = df['label_id'].values

    # Sınıf Ağırlıkları
    class_counts = Counter ( y )
    total_samples = len ( y )
    num_classes = len ( Config.LABEL_MAP )

    class_weights = []
    for i in range ( num_classes ):
        count = class_counts.get ( i , 0 )
        weight = total_samples / (num_classes * count) if count > 0 else 1.0
        class_weights.append ( weight )

    weights_tensor = torch.FloatTensor ( class_weights ).to ( device )

    # Dataset
    X_tensor = torch.tensor ( X , dtype = torch.long )
    y_tensor = torch.tensor ( y , dtype = torch.long )
    dataset = EmotionDataset ( X_tensor , y_tensor )
    train_loader = DataLoader ( dataset , batch_size = Config.BATCH_SIZE , shuffle = True )

    print ( "\n--- 2. Model Başlatılıyor ---" )
    from core.model import EmotionNN
    model = EmotionNN ( len ( word2idx ) , Config.EMBED_DIM , Config.HIDDEN_DIM , len ( Config.LABEL_MAP ) )
    model.to ( device )

    # --- KRİTİK DEĞİŞİKLİK BURADA ---
    # label_smoothing=0.1 ekledik. Modelin "aşırı emin" olmasını engeller.
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

    if not os.path.exists ( "models" ):
        os.makedirs ( "models" )
    torch.save ( model.state_dict () , "models/emotion_model.pth" )
    print ( "\n--- Model Label Smoothing ile Kaydedildi! ---" )


if __name__ == "__main__":
    train_pipeline ()