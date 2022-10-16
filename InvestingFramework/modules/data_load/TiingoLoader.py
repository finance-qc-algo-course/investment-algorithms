import re
import requests
import pandas as pd
import os

class TiingoLoader:
    API_REQUEST_TEMPLATE = "https://api.tiingo.com/tiingo/daily/{TICKER}/prices?startDate={START_DATE}&endDate={END_DATE}"

    def __init__(self, api_token_path: None):
        if api_token_path is None:
            raise ValueError("API token path is not given")
        
        self.api_token = self.read_token(api_token_path)

    def read_token(self, api_token_path: str):
        with open(api_token_path, 'r') as api_token_file:
            return api_token_file.read()

    def load_data(self, ticker: str, date_from: str, date_to: str):
        request = self.API_REQUEST_TEMPLATE
        request = re.sub("\{TICKER\}", ticker, request)
        request = re.sub("\{START_DATE\}", date_from, request)
        request = re.sub("\{END_DATE\}", date_to, request)

        headers = {
            "Content-Type": "application/json",
            "Authorization" : f"Token {self.api_token}"
        }

        request_result = requests.get(request, headers=headers)
        print(request_result)

        data = pd.DataFrame(request_result.json())
        data["date"] = data.date.apply(lambda x: pd.Timestamp(x).to_datetime64())
        data = data.set_index("date")
        data = data[["close", "high", "low", "open", "volume"]]
        data.columns = ["Close", "High", "Low", "Open", "Volume"]

        return data