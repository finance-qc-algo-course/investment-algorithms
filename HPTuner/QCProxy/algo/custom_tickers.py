# Manual QC500 model

# GOOG -> GOOGV, GOOGL ?
def get_custom_top36_tickers():
    return [
        'EXPD', 'GILD', 'FAST', 'ADSK', 'ATVI', 'KLAC',
        'FISV', 'CSCO', 'MSFT', 'CTSH', 'CHKP', 'PAYX',
        'AAPL', 'BBBY', 'COST', 'CHRW', 'WYNN', 'ADBE',
        'SBUX', 'QCOM', 'NTAP', 'VRTX', 'AMAT', 'PCAR',
        'ISRG', 'EBAY', 'NVDA', 'GRMN', 'AMZN', 'INTC',
        'CTXS', 'INTU', 'BIIB', 'ESRX', 'CMCSA', 'AMGN'
    ]


def get_custom_top_tickers(n: int):
    if n < 1 or n > 36:
        raise ValueError('can get 1 to 36 tickers, not {}'.format(n))

    return get_custom_top36_tickers()[:n]
