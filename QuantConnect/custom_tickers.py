# Manual QC500 model

# def get_custom_top46_tickers():
#     return ['BRCM', 'EXPD', 'GILD', 'FAST', 'ADSK', 'ATVI', 'KLAC', 'FISV', 'GOOG', 'CSCO', 'ALTR', 'SIAL', 'YHOO', 'MSFT', 'CTSH', 'CHKP', 'XLNX', 'PAYX', 'AAPL', 'BBBY', 'COST', 'CHRW', 'WYNN', 'ADBE', 'SBUX', 'QCOM', 'NTAP', 'LLTC', 'VRTX', 'AMAT', 'CELG', 'SPLS', 'PCAR', 'ISRG', 'EBAY', 'NVDA', 'GRMN', 'AMZN', 'INTC', 'CTXS', 'INTU', 'BIIB', 'SYMC', 'ESRX', 'CMCSA', 'AMGN']

# GOOG -> GOOGV, GOOGL ?
def get_custom_top45_tickers():
    return [ \
        'BRCM', 'EXPD', 'GILD', 'FAST', 'ADSK', 'ATVI', 'KLAC', 'FISV', \
        'CSCO', 'ALTR', 'SIAL', 'YHOO', 'MSFT', 'CTSH', 'CHKP', 'XLNX', \
        'PAYX', 'AAPL', 'BBBY', 'COST', 'CHRW', 'WYNN', 'ADBE', 'SBUX', \
        'QCOM', 'NTAP', 'LLTC', 'VRTX', 'AMAT', 'CELG', 'SPLS', 'PCAR', \
        'ISRG', 'EBAY', 'NVDA', 'GRMN', 'AMZN', 'INTC', 'CTXS', 'INTU', \
        'BIIB', 'SYMC', 'ESRX', 'CMCSA', 'AMGN'\
        ] # TODO + ['GOOG']
    
def get_custom_top_tickers(n: int):
    if n < 1 or n > 45:
        raise ValueError('can get 1 to 45 tickers, not {}'.format(n))
        
    return get_custom_top45_tickers()[:n]

