# data_loader.py
import kagglehub
import pandas as pd
import os
import torch
from torch.utils.data import Dataset , DataLoader
from core.config import Config
from utils import clean_text , text_to_indices


class EmotionDataset ( Dataset ):
    def __init__(self , X , y):
        self.X = X
        self.y = y

    def __len__(self): return len ( self.X )

    def __getitem__(self , idx): return self.X[idx] , self.y[idx]


def get_data_loaders():
    print ( "--- Veri Hazırlanıyor ---" )
    path = kagglehub.dataset_download ( Config.DATASET_URL )
    csv_file = [f for f in os.listdir ( path ) if f.endswith ( '.csv' )][0]
    df = pd.read_csv ( os.path.join ( path , csv_file ) )

    # Sütun düzeltme ve Label Mapping (Önceki kodundaki mantık)
    df.columns = [c.lower () for c in df.columns]
    # ... (Buraya önceki cevaptaki sütun düzeltme kodlarını ekle) ...

    # Basitleştirilmiş haliyle:
    if 'text' not in df.columns: df.rename ( columns = {'content': 'text'} , inplace = True )

    # Vocab Oluşturma
    word2idx = {"<PAD>": 0 , "<UNK>": 1}
    idx = 2
    for text in df['text']:
        cleaned = clean_text ( text )
        for word in cleaned.split ():
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1

    # Veriyi Sayısallaştırma
    X = [text_to_indices ( t , word2idx , Config.MAX_LEN ) for t in df['text']]
    # Label işlemleri... (df['label_id'] oluşturduğunu varsayıyoruz)
    # ...

    # Tensor Dönüşümü
    X_tensor = torch.tensor ( X , dtype = torch.long )
    y_tensor = torch.tensor ( df['label_id'].values , dtype = torch.long )  # label_id olduğundan emin ol

    train_ds = EmotionDataset ( X_tensor , y_tensor )
    train_loader = DataLoader ( train_ds , batch_size = Config.BATCH_SIZE , shuffle = True )

    return train_loader , word2idx