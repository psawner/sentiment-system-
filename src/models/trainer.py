from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

class ModelTrainer:
    def __init__(self):
        self.model = None

    def model_architecture(self, tokenizer, maxlen, le):
        vocab_size = tokenizer.num_words 

        embedding_dim = 128
        num_classes = len(le.classes_)

        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    
    def complie(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train_test(self, X_train_padded, df_train_target, X_val_padded, df_test_target):
        return self.model.fit(
            X_train_padded, df_train_target,
            epochs=10, 
            batch_size=128, 
            validation_data=(X_val_padded, df_test_target)
        )
        
    
    def model_save(self):
        model_path = "artifacts/sentiment_model.keras"
        self.model.save(model_path)
        return model_path

