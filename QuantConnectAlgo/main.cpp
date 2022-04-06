from datetime import timedelta

class BullCallSpreadAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 4, 1)
        self.SetEndDate(2017, 6, 30)
        self.SetCash(1000000)
        equity = self.AddEquity("GOOG", Resolution.Minute)
        option = self.AddOption("GOOG", Resolution.Minute)
        self.symbol = option.Symbol
        option.SetFilter(-6, 6, timedelta(30), timedelta(60))
        self.SetBenchmark(equity.Symbol)
        
    def OnData(self,slice):
        for i in slice.OptionChains:
            chains = i.Value
            if not self.Portfolio.Invested: 
                self.TradeOptions(chains) 
 
    def TradeOptions(self,chains):
        # sorted the optionchain by expiration date and choose the furthest date
        expiry = sorted(chains,key = lambda x: x.Expiry, reverse=True)[0].Expiry
        # filter the call and put contract
        call = [i for i in chains if i.Expiry == expiry and i.Right == OptionRight.Call]
        put = [i for i in chains if i.Expiry == expiry and i.Right == OptionRight.Put]
        
        # sorted the contracts according to their strike prices 
        call_contracts = sorted(call,key = lambda x: x.Strike)    
        if len(call_contracts) == 0: return
        self.call = call_contracts[0]
        for i in put:
            if i.Strike == self.call.Strike:
                self.put = i
        self.Buy(self.call.Symbol, 1)
        self.Buy(self.put.Symbol ,1)
    
    def OnOrderEvent(self, orderEvent):
        self.Log(str(orderEvent))
    

class IronButterflyAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 4, 1)
        self.SetEndDate(2017, 4, 30)
        self.SetCash(10000000)
        equity = self.AddEquity("GOOG", Resolution.Minute)
        option = self.AddOption("GOOG", Resolution.Minute)
        self.symbol = option.Symbol
        option.SetFilter(-10, 10, timedelta(0), timedelta(30))
        # use the underlying equity GOOG as the benchmark
        self.SetBenchmark(equity.Symbol)
        
    def OnData(self,slice):

        if self.Portfolio["GOOG"].Quantity != 0:
            self.Liquidate()
        if not self.Portfolio.Invested and self.Time.hour != 0 and self.Time.minute != 0: 
            for i in slice.OptionChains:
                if i.Key != self.symbol: continue
                chain = i.Value
                contract_list = [x for x in chain]
                # if there is no optionchain or no contracts in this optionchain, pass the instance
                if (slice.OptionChains.Count == 0) or (len(contract_list) == 0): 
                    return  
                # sorted the optionchain by expiration date and choose the furthest date
                expiry = sorted(chain,key = lambda x: x.Expiry)[-1].Expiry
                # filter the call and put options from the contracts
                call = [i for i in chain if i.Right == 0 and i.Expiry == expiry]
                put = [i for i in chain if i.Right == 1 and  i.Expiry == expiry]
                # sorted the contracts according to their strike prices 
                call_contracts = sorted(call,key = lambda x: x.Strike)    
                put_contracts = sorted(put,key = lambda x: x.Strike)    
                if len(call_contracts) == 0 or len(put_contracts) == 0 : continue
                # Sell 1 ATM Put 
                atm_put = sorted(put_contracts,key = lambda x: abs(chain.Underlying.Price - x.Strike))[0]
                self.Sell(atm_put.Symbol ,1)
                # Sell 1 ATM Call
                atm_call = sorted(call_contracts,key = lambda x: abs(chain.Underlying.Price - x.Strike))[0]
                self.Sell(atm_call.Symbol ,1)
                # Buy 1 OTM Call 
                otm_call = call_contracts[-1]
                self.Buy(otm_call.Symbol ,1)
                # Buy 1 OTM Put 
                otm_put = put_contracts[0]
                self.Buy(otm_put.Symbol ,1)
                
                self.trade_contracts = [atm_put, atm_call, otm_call, otm_put]
    
    def OnOrderEvent(self, orderEvent):
        self.Log(str(orderEvent))
