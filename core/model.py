# core/model.py
"""
Emotion Recognition Model Architecture
--------------------------------------
This module defines the neural network architecture used for emotion classification.
It implements a custom CNN (Convolutional Neural Network) optimized for text classification tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Config import etmeye gerek kalmadı çünkü parametre olarak alacağız,
# ama yine de dursun zararı yok.
from core.config import Config

# Sınıf adını EmotionNN -> EmotionCNN olarak değiştirdik
class EmotionCNN(nn.Module):
    # __init__ kısmına 'dropout' parametresini ekledik, çünkü predictor bunu gönderiyor.
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.5):
        super(EmotionCNN, self).__init__()

        # 1. Embedding (Kelimeleri sayıya dökme)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. Convolutional Layers (Metindeki duygu kalıplarını yakalar)
        # Farklı pencere boyutlarıyla (3'lü, 4'lü, 5'li kelime grupları) tarama yapar
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=5, padding=2)

        # 3. Dropout (Ezber bozma)
        # Parametre olarak gelen dropout değerini kullanıyoruz
        self.dropout = nn.Dropout(dropout)

        # 4. Fully Connected (Karar verme)
        # 3 farklı conv katmanından gelen özellikleri birleştiriyoruz (hidden_dim * 3)
        self.fc = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, text):
        # text: [batch_size, seq_len]

        # Embedding
        embedded = self.embedding(text)  # [batch, seq_len, embed_dim]

        # Conv1d için boyut değişimi gerekiyor: [batch, embed_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # Convolution + ReLU + MaxPool
        # 3 farklı filtreyle tarıyoruz
        x1 = F.relu(self.conv1(embedded))
        # Max pool boyutunu dinamik alıyoruz ki hata vermesin
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)

        x2 = F.relu(self.conv2(embedded))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)

        x3 = F.relu(self.conv3(embedded))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)

        # Özellikleri birleştir
        combined = torch.cat((x1, x2, x3), dim=1)

        # Dropout ve Sınıflandırma
        combined = self.dropout(combined)
        output = self.fc(combined)

        return output