import pandas as pd

class Model:
    def __init__(self, train_data):
        X_train = self.preprocess(train_data)
        y_train = self.generate_target(train_data)
        self.train(X_train, y_train)

    def generate_target(self, X_train):
        raise NotImplementedError()

    def train(self, X_train: pd.DataFrame, y_train):
        raise NotImplementedError()

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame):
        raise NotImplementedError()

    def make_trade(self, data: pd.DataFrame):
        preprocessed_data = self.preprocess(data)
        return self.predict(preprocessed_data)

