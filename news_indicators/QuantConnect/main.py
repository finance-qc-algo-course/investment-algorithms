from datetime import timedelta
import model_stuff


FEE = 0.001

class CustomFeeModel:
    def GetOrderFee(self, parameters):
        fee = parameters.Security.Price * parameters.Order.AbsoluteQuantity * FEE
        return OrderFee(CashAmount(fee, 'USD'))


class Algo(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2021, 8, 7)
        self.SetEndDate(2022, 5, 1)
        self.SetCash(200000)
        
        self.crypto = self.AddCrypto("BTCUSD", Resolution.Minute, Market.GDAX)
        self.crypto.SetFeeModel(CustomFeeModel())
        self.SetBenchmark(self.crypto.Symbol)
        
        self.my_model = model_stuff.Model()
        self.slice = None
        
        self.bought = False
        self.sold = False
        
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=15)), self.Trade)

    def OnData(self, slice):
        self.slice = slice

    def Trade(self):
        if self.bought:
            self.bought = False
            self.Sell(self.crypto.Symbol, 1)
        if self.sold:
            self.sold = False
            self.Buy(self.crypto.Symbol, 1)
        
        history = self.History(self.crypto.Symbol, 15 * 48 + 30, Resolution.Minute)
        history = history.reset_index()
        history = history.set_index('time')[['close', 'high', 'low', 'open', 'volume']]
        
        try:
            pred_profit = self.my_model.predict(history)
            if pred_profit > 2 * FEE:
                self.bought = True
                self.Buy(self.crypto.Symbol, 1)
            elif pred_profit < -2 * FEE:
                self.sold = True
                self.Sell(self.crypto.Symbol, 1)
        except:
            pass

    
    def OnOrderEvent(self, orderEvent):
        self.Log(str(orderEvent))
 
