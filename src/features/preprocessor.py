import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib

class DataPreprocessor:

    def clean_data(self, df_train, df_test):
        df_train = df_train.dropna()
        df_test = df_test.dropna()
        return df_train, df_test

    def clean_text(self, text):
        if not isinstance(text, str): 
            return ""
    
        #Lowercasing
        text = text.lower()

        #Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def encoding_target(self, df_train, df_test):
        le = LabelEncoder()

        df_train['encoded_sentiment'] = le.fit_transform(df_train['Sentiment'])
        df_test['encoded_sentiment'] = le.transform(df_test['Sentiment'])
        
        #save labels
        joblib.dump(le.classes_, "artifacts/label_classes.pkl")

        return df_train['encoded_sentiment'], df_test['encoded_sentiment'], le
    
    def processed_text(self, df_train, df_test):
        df_train['processed_content_nn'] = df_train['Tweet_Content'].apply(self.clean_text)
        df_test['processed_content_nn'] = df_test['Tweet_Content'].apply(self.clean_text)
        return df_train, df_test
    
    def text_preprocessing(self, df_train, df_test):
        tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")

        tokenizer.fit_on_texts(df_train['processed_content_nn'])


        X_train_sequences = tokenizer.texts_to_sequences(df_train['processed_content_nn'])
        X_val_sequences = tokenizer.texts_to_sequences(df_test['processed_content_nn'])

        maxlen = max(len(x) for x in X_train_sequences)

        #save maxlen and tokenizer
        joblib.dump(tokenizer, "artifacts/tokenizer.pkl")
        joblib.dump(maxlen, "artifacts/max_len.pkl")

        X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding='post')
        X_val_padded = pad_sequences(X_val_sequences, maxlen=maxlen, padding='post')

        return tokenizer, maxlen, X_train_padded, X_val_padded
    



    
