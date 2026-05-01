# src/models/predictor.py

import numpy as np
import pickle
import joblib
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Predictor:
    def __init__(self, model_path, tokenizer_path, max_len_path, label_path):
        self.model = load_model(model_path)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.max_len = joblib.load(max_len_path)
        self.label_classes = joblib.load(label_path)

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def predict(self, text):
        cleaned = self.preprocess(text)

        seq = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')

        prediction = self.model.predict(padded)

        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        label = self.label_classes[predicted_class]

        return label, confidence, prediction[0]