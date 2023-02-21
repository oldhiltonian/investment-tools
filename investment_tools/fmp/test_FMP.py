import unittest
import requests
import pandas as pd
import numpy as np
import datetime as dt
from fmp_layer_1 import FinancialData, ManualAnalysis, Plots, Company
from pathlib import Path
import itertools

key_path = Path().home()/'desktop'/'FinancialModellingPrep_API.txt'
with open(key_path) as file:
    api_key = file.read()


class TestFinancialData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # self.tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
        self.tickers = ['AAPL']
        self.data =    ['online', 'local']
        self.period =  ['annual', 'quarter']
        self.api_key = api_key
        self.limit = 120
        self.zipped_args_tdp = list(itertools.product(self.tickers, self.data, self.period))

    def test_assert_valid_user_input(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            instance.assert_valid_user_inputs()
        
    def test_generate_request_url(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            bs_str = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
            is_str = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
            cfs_str = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
            metric_str = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
            self.assertEqual(instance.generate_request_url('bs'), bs_str)
            self.assertEqual(instance.generate_request_url('is'), is_str)
            self.assertEqual(instance.generate_request_url('cfs'), cfs_str)
            self.assertEqual(instance.generate_request_url('metrics'), metric_str)
            with self.assertRaises(ValueError):
                instance.generate_request_url('')
                instance.generate_request_url(4)
                instance.generate_request_url('42')


    def test_fetch_raw_data(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = FinancialData(ticker, api_key, data, period)
            for string in ['bs', 'is', 'cfs', 'metrics']:
                raw_data = instance.fetch_raw_data(string)
                expected_type = requests.Response if data == 'online' else pd.DataFrame
                self.assertEqual(isinstance(raw_data, expected_type), True)
                with self.assertRaises(ValueError):
                    instance.fetch_raw_data('')
                    instance.fetch_raw_data(4)
                    instance.fetch_raw_data(42)
        


if __name__ == '__main__':

    unittest.main()
    