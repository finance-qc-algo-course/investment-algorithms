from backtesting.lib import SignalStrategy
from .Model import Model
from ...preprocess.Preprocessor import Preprocessor

class ModelAlgorithm(SignalStrategy):
    def init(self, model: Model = None, data_preprocessor: Preprocessor = None, sequence_len: int = None):
        super().init()

        self.model = model
        self.preprocessor = data_preprocessor
        self.sequence_len = sequence_len

    def next(self):
        if self.model.predict(self.preprocessor(self.data[-self.sequence_len:])) > 0:
            self.buy()