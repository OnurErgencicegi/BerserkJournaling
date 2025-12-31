# core/config.py
class Config:
    DATASET_URL = "parulpandey/emotion-dataset"
    MAX_LEN = 50  # 35'ten 50'ye çıkardık (Uzun cümleleri kesmesin)
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