from src.data.data_loader import DataLoader
from src.features.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import Evaluator

class TrainingPipeline:
    def __init__(self):
        self.loader = DataLoader("data/raw/twitter_training.csv", "data/raw/twitter_validation.csv")
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()


    def run(self):
        #loading training and testing data
        df_train, df_test = self.loader.load_data()

        #preprocessing
        #1) cleaning
        df_train, df_test = self.preprocessor.clean_data(df_train, df_test)

        #2) processed_text
        df_train, df_test = self.preprocessor.processed_text(df_train, df_test)

        #3) encoding target
        df_train_target, df_test_target, le = self.preprocessor.encoding_target(df_train, df_test)

        #4) tetxt_preprocessing
        tokenizer, maxlen, X_train_padded, X_val_padded = self.preprocessor.text_preprocessing(df_train, df_test)

        #model training
        #1) model architechture
        self.trainer.model_architecture(tokenizer, maxlen, le)

        #2) compile
        self.trainer.complie()

        #3) training
        self.trainer.train_test(X_train_padded, df_train_target, X_val_padded, df_test_target)

        #4) model save
        model_path = self.trainer.model_save()

        #model evaluation
        self.evaluator = Evaluator(model_path)
        self.evaluator.evaluate(X_val_padded, df_test)
        self.evaluator.classification_report(X_val_padded, df_test, le)
