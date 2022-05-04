from datetime import timedelta
import model_stuff

class LongStraddleAndIronButterflyAlgorithm(QCAlgorithm):

    def Initialize(self):
        # self.SetStartDate(2017, 4, 1)   #Super, but expected
        # self.SetEndDate(2017, 6, 30)
        # self.SetStartDate(2021, 10, 1)  #Ok
        # self.SetEndDate(2022, 1, 30)
        # self.SetStartDate(2021, 4, 1)   #Super
        # self.SetEndDate(2021, 8, 30)
        self.SetStartDate(2022, 4, 1)     #Bad, but it's ok
        self.SetEndDate(2022, 4, 30)
        self.SetCash(1000000)
        equity = self.AddEquity("GOOG", Resolution.Minute)
        option = self.AddOption("GOOG", Resolution.Minute)
        self.symbol = option.Symbol
        option.SetFilter(-6, 6, timedelta(30), timedelta(60))
        self.SetBenchmark(equity.Symbol)
        
        self.my_model = model_stuff.Model()
        self.slice = None
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=30)), self.Trade)

    def OnData(self, slice):
        self.slice = slice

    def Trade(self):
        if self.slice is None:
            return
        if self.Portfolio["GOOG"].Quantity != 0:
            self.Liquidate()
        hist = self.History(self.Symbol("GOOG"), 370, Resolution.Minute)
        if self.slice.Bars.ContainsKey("GOOG"):
            for i in self.slice.OptionChains:
                chains = i.Value
                if self.Time.hour != 0 and self.Time.minute != 0:
                    prediction = self.my_model.predict_valotile(hist)
                    self.Debug('Prediction is: ' + str(prediction))
                    if prediction > 0.8:
                        self.Debug("Long Straddle")
                        self.LongStraddleTradeOptions(chains) 
                    elif prediction < 0.2:
                        self.Debug("Iron Butterfly")
                        # self.IronButterflyTradeOptions(chains) 
                        # self.Debug("Iron Condor")
                        # self.IronCondorTradeOprions(chains) 
 
    def LongStraddleTradeOptions(self, chains):
        # sorted the optionchain by expiration date and choose the furthest date
        expiry = sorted(chains,key = lambda x: x.Expiry)[0].Expiry
        # filter the call and put contract
        call = [i for i in chains if i.Expiry == expiry and i.Right == OptionRight.Call]
        put = [i for i in chains if i.Expiry == expiry and i.Right == OptionRight.Put]
        
        # sorted the contracts according to their strike prices 
        call_contracts = sorted(call,key = lambda x: x.Strike)    
        if len(call_contracts) == 0: return
        self.call = call_contracts[0]
        
        put_contracts = sorted(put,key = lambda x: x.Strike)    
        if len(put_contracts) == 0: return
        self.put = call_contracts[0]
        
        self.Buy(self.call.Symbol, 1)
        self.Buy(self.put.Symbol ,1)
    
    def IronButterflyTradeOptions(self, chain):
        contract_list = [x for x in chain]
        # if there is no optionchain or no contracts in this optionchain, pass the instance
        if (len(contract_list) == 0): 
            return  
        # sorted the optionchain by expiration date and choose the furthest date
        expiry = sorted(chain,key = lambda x: x.Expiry)[0].Expiry
        # filter the call and put options from the contracts
        call = [i for i in chain if i.Right == 0 and i.Expiry == expiry]
        put = [i for i in chain if i.Right == 1 and  i.Expiry == expiry]
        # sorted the contracts according to their strike prices 
        call_contracts = sorted(call,key = lambda x: x.Strike)    
        put_contracts = sorted(put,key = lambda x: x.Strike)    
        if len(call_contracts) == 0 or len(put_contracts) == 0 : return
         
        atm_put = sorted(put_contracts,key = lambda x: abs(chain.Underlying.Price - x.Strike))[0]
        atm_call = sorted(call_contracts,key = lambda x: abs(chain.Underlying.Price - x.Strike))[0]
        otm_call = call_contracts[-1]
        otm_put = put_contracts[0]
              
        self.Sell(atm_put.Symbol, 1)
        self.Sell(atm_call.Symbol, 1) 
        self.Buy(otm_call.Symbol, 1)
        self.Buy(otm_put.Symbol, 1)
        
    def IronCondorTradeOprions(self, chain):
        contract_list = [x for x in chain]
        # if there is no optionchain or no contracts in this optionchain, pass the instance
        if (len(contract_list) == 0): 
            return   
        # sorted the optionchain by expiration date and choose the furthest date
        expiry = sorted(chain,key = lambda x: x.Expiry)[0].Expiry
        # filter the call and put options from the contracts
        call = [i for i in chain if i.Expiry == expiry and i.Right == 0]
        put = [i for i in chain if i.Expiry == expiry and i.Right == 1]
        # sorted the contracts according to their strike prices 
        call_contracts = sorted(call,key = lambda x: x.Strike)    
        put_contracts = sorted(put,key = lambda x: x.Strike)    
        if len(call_contracts) < 10 or len(put_contracts) < 10: return 
        otm_put_lower = put_contracts[0]
        otm_put = put_contracts[9]
        otm_call = call_contracts[-10]
        otm_call_higher = call_contracts[-1]
        self.trade_contracts = [otm_call.Symbol,otm_call_higher.Symbol,otm_put.Symbol,otm_put_lower.Symbol]
    
        # if there is no securities in portfolio, trade the options 
        self.Buy(otm_put_lower.Symbol ,1)
        self.Sell(otm_put.Symbol ,1)
        self.Sell(otm_call.Symbol ,1)
        self.Buy(otm_call_higher.Symbol ,1)
    
    def OnOrderEvent(self, orderEvent):
        self.Log(str(orderEvent))
