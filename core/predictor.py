# core/predictor.py
import os
import torch
import torch.nn.functional as F
from transformers import pipeline  # GoogleTranslator yerine bunu kullanıyoruz

# Kendi modüllerin
from core.config import Config
from core.utils import text_to_indices , get_word2idx
from core.model import EmotionCNN


class MoodPredictor:
    def __init__(self):
        self.device = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
        self.config = Config ()

        # 1. Sözlüğü Yükle
        self.word2idx = get_word2idx ()

        # 2. Duygu Modelini Başlat
        self.model = EmotionCNN (
            vocab_size = len ( self.word2idx ) + 1 ,
            embed_dim = self.config.EMBED_DIM ,
            hidden_dim = self.config.HIDDEN_DIM ,
            output_dim = len ( self.config.LABEL_MAP ) ,
            dropout = self.config.DROPOUT
        ).to ( self.device )

        # 3. Duygu Modeli Ağırlıklarını Yükle
        base_dir = os.path.dirname ( os.path.dirname ( os.path.abspath ( __file__ ) ) )
        model_path = os.path.join ( base_dir , "models" , "emotion_model.pth" )

        try:
            self.model.load_state_dict ( torch.load ( model_path , map_location = self.device ) )
        except FileNotFoundError:
            print (
                f"UYARI: {model_path} bulunamadı. Model rastgele ağırlıklarla çalışıyor! (Önce train.py çalıştırın)" )
            pass
        except Exception as e:
            print ( f"Duygu modeli yükleme hatası: {e}" )

        self.model.eval ()

        # Label map ters çevir (0 -> sadness)
        self.idx2label = {v: k for k , v in self.config.LABEL_MAP.items ()}

        # 4. OFFLINE ÇEVİRİ MODELİNİ YÜKLE (Helsinki-NLP)
        # Bu kısım internet olmadan çalışmasını sağlar.
        print ( "Çeviri modeli yükleniyor... (Helsinki-NLP/opus-mt-tr-en)" )
        try:
            self.translator = pipeline ( "translation" , model = "Helsinki-NLP/opus-mt-tr-en" )
        except Exception as e:
            print ( f"Çeviri modeli yüklenemedi: {e}" )
            self.translator = None

    def predict(self , text_tr):
        """
        Türkçe metni alır -> Offline çevirir -> Duygu ve Olasılıkları döndürür.
        """
        # 1. Çeviri (Offline)
        translated_text = text_tr

        if self.translator:
            try:
                # Metin çok uzunsa model çökmesin diye ilk 512 karakteri alıyoruz.
                # Duygu analizi için bu kadarı genelde yeterlidir.
                text_to_translate = text_tr[:512]

                result = self.translator ( text_to_translate )
                translated_text = result[0]['translation_text']
            except Exception as e:
                print ( f"Çeviri hatası (Offline): {e}" )
                # Hata olursa (örn. çok garip karakterler) orijinal metinle devam et

        # 2. Tokenize (Sayıya çevir)
        indices = text_to_indices ( translated_text , self.word2idx , self.config.MAX_LEN )
        tensor_input = torch.LongTensor ( [indices] ).to ( self.device )

        # 3. Tahmin
        with torch.no_grad ():
            outputs = self.model ( tensor_input )
            probs_tensor = F.softmax ( outputs , dim = 1 )  # Olasılığa çevir

            # En yüksek sınıf
            top_prob , top_class = torch.max ( probs_tensor , 1 )
            predicted_label = self.idx2label[top_class.item ()]

            # Tüm olasılıklar sözlüğü
            probs_dict = {}
            for idx , prob in enumerate ( probs_tensor[0] ):
                if idx in self.idx2label:
                    label_name = self.idx2label[idx]
                    probs_dict[label_name] = prob.item ()

        return predicted_label , probs_dict