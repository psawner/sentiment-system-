import pandas as pd

class DataLoader:
    def __init__(self, file_path1, file_path2):
        self.file_path1 = file_path1
        self.file_path2 = file_path2

    def load_data(self):
        column_names = ['Tweet_ID', 'Entity', 'Sentiment', 'Tweet_Content']
        df_train = pd.read_csv(self.file_path1, names=column_names)
        df_test = pd.read_csv(self.file_path2,names=column_names)
        return df_train, df_test