# core/predictor.py
import torch
from core.config import Config
from core.utils import text_to_indices , get_word2idx  # get_word2idx'i buraya ekledik
from core.model import EmotionNN
from deep_translator import GoogleTranslator


class MoodPredictor:
    def __init__(self , model_path="models/emotion_model.pth"):
        self.device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )

        # 1. Önce Sözlüğü Yükle (Boyutu öğrenmek için)
        self.word2idx = get_word2idx ()
        vocab_size = len ( self.word2idx )  # Dinamik boyut hesabı

        # 2. Modeli Başlat (Config.VOCAB_SIZE yerine hesaplanan boyutu kullan)
        self.model = EmotionNN ( vocab_size , Config.EMBED_DIM , Config.HIDDEN_DIM , len ( Config.LABEL_MAP ) )

        # 3. Eğitilmiş Ağırlıkları Yükle
        try:
            self.model.load_state_dict ( torch.load ( model_path , map_location = self.device ) )
        except FileNotFoundError:
            print ( f"Uyarı: Model dosyası bulunamadı ({model_path}). Lütfen önce eğitim yapın." )
            pass

        self.model.to ( self.device )
        self.model.eval ()

    def predict(self , text_tr):
        # Çeviri
        try:
            translated = GoogleTranslator ( source = 'tr' , target = 'en' ).translate ( text_tr )
        except:
            return None , "Çeviri Hatası"

        # Hazırlık (self.word2idx kullanıyoruz artık)
        indices = text_to_indices ( translated , self.word2idx , Config.MAX_LEN )
        tensor_input = torch.tensor ( [indices] , dtype = torch.long ).to ( self.device )

        # Tahmin
        with torch.no_grad ():
            logits = self.model ( tensor_input )
            probs = torch.nn.functional.softmax ( logits , dim = 1 )[0].tolist ()

        idx2label = {v: k for k , v in Config.LABEL_MAP.items ()}
        result = {idx2label[i]: p for i , p in enumerate ( probs ) if i in idx2label}

        dominant_mood = max ( result , key = result.get )
        return dominant_mood , result