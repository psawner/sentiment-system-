from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np

class Evaluator:
    def __init__(self, model_path):
        self.loaded_model = tf.keras.models.load_model(model_path)

    def evaluate(self, X_val_padded, df_test):
        loss, accuracy = self.loaded_model.evaluate(X_val_padded, df_test['encoded_sentiment'], verbose=0)
        
        return f"model accuracy on test data: {accuracy*100:.2f}%\nmodel loss on test data: {loss}"
    
    def model_summary(self):
        return self.loaded_model.summary()
    
    def classification_report(self, X_val_padded, df_test, le):
        y_pred_ann_raw = self.loaded_model.predict(X_val_padded,verbose=0)
        y_pred_ann = np.argmax(y_pred_ann_raw, axis=1)

        return classification_report(df_test['encoded_sentiment'], y_pred_ann, target_names=le.classes_)