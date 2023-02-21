import unittest
import requests
import pandas as pd
import numpy as np
import datetime as dt
from fmp_layer_1 import FinancialData, ManualAnalysis, Plots, Company
from pathlib import Path

key_path = Path().home()/'desktop'/'FinancialModellingPrep_API.txt'
with open(key_path) as file:
    api_key = file.read()


'''If you only want to do something on the class itself, add the @classmethod before the setup .
    This wills top it recalling the API every time it runs new tests

    @classmethod
    def setUpClass(self):
        self.data = API call
    
    It's also looking like each class method will require its own test class'''

class TestFinancialData(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tickers = ['AAPL', 'MSFT', 'NVDA', 'VAC', 'WBA']
        self.api_key = api_key
        self.limit = 10
        self.data = ['online', 'local', 'local', 'online', 'local']
        self.period = ['annual', 'quarter', 'annual', 'annual', 'quarter']
        self.zipped_args = zip(self.tickers, self.data, self.period)
        self.instance_annual = FinancialData('AAPL', self.api_key, 'online', period='annual', limit=10)
        self.instance_quarter = FinancialData('AAPL', self.api_key, 'online', period='quarter', limit=10)
    
        
    def test_assert_valid_user_input(self):
        for ticker, data, period in self.zipped_args:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            instance.assert_valid_user_inputs()
        
    def test_generate_request_url(self):
        for ticker, data, period in self.zipped_args:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            bs_str = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}'
            is_str = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}'
            cfs_str = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={api_key}'
            metric_str = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={period}&limit={limit}&apikey={api_key}'
            self.assertEqual(instance.generate_request_url('bs'), bs_str)
            self.assertEqual(instance.generate_request_url('is'), is_str)
            self.assertEqual(instance.generate_request_url('cfs'), cfs_str)
            self.assertEqual(instance.generate_request_url('metrics'), metric_str)
            self.assertRaises(instance.generate_request_url(''), ValueError)
            self.assertRaises(instance.generate_request_url(4), ValueError)
            self.assertRaises(instance.generate_request_url('42'), ValueError)


    def test_fetch_raw_data(self):
        for ticker, data, period in self.zipped_args:
            instance = FinancialData(ticker, api_key, data, period)
            for string in ['bs', 'is', 'cfs', 'metric', '']:
                raw_data = instance.fetch_raw_data(string)
                self.assertEqual(isinstance(raw_data, requests.Response), True)
        
    # def test_get_load_path(self):
    #     bs_path = self.data.get_load_path('bs', 'AAPL', 'annual')
    #     expected = 'C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\Company Financial Data\\AAPL\\annual\\balance_sheets.parquet'
    #     self.assertEqual(str(bs_path), expected)


if __name__ == '__main__':

    unittest.main()
    