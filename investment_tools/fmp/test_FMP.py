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
    def setUpClass(cls):
        # self.tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
        cls.tickers = ['AAPL']
        cls.api_key = api_key
        cls.data =    ['online', 'local']
        cls.period =  ['annual', 'quarter']
        cls.limit = 120
        cls.zipped_args_tdp = list(itertools.product(cls.tickers, cls.data, cls.period))



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
        
    def test_get_load_path(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            bs_str = (f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
                            Company_Financial_Data\\{ticker}\\{period}\\balance_sheets.parquet''')
            is_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
                            Company_Financial_Data\\{ticker}\\{period}\\income_statements.parquet'''
            cfs_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
                        Company_Financial_Data\\{ticker}\\{period}\\cash_flow_statements.parquet'''
            metric_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
                        Company_Financial_Data\\{ticker}\\{period}\\reported_key_metrics.parquet'''
            price_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
                        Company_Financial_Data\\{ticker}\\{period}\\stock_price_data.parquet'''
            bs_str = bs_str.replace('\n', '').replace('\t', '').replace(' ', '')
            is_str = is_str.replace('\n', '').replace('\t', '').replace(' ', '')
            cfs_str = cfs_str.replace('\n', '').replace('\t', '').replace(' ', '')
            metric_str = metric_str.replace('\n', '').replace('\t', '').replace(' ', '')
            price_str = price_str.replace('\n', '').replace('\t', '').replace(' ', '')

            self.assertEqual(str(Path(bs_str)), str(instance.get_load_path('bs', ticker, period)))
            self.assertEqual(str(Path(is_str)), str(instance.get_load_path('is', ticker, period)))
            self.assertEqual(str(Path(cfs_str)), str(instance.get_load_path('cfs', ticker, period)))
            self.assertEqual(str(Path(metric_str)), str(instance.get_load_path('metrics', ticker, period)))
            self.assertEqual(str(Path(price_str)), str(instance.get_load_path('price', ticker, period)))







# class MyClass:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
        
#     def mult_(self):
#         return self.a* self.b
    
#     def add_(self):
#         return self.a + self.b

#     def div_(self):
#         return self.a/self.b

#     def sub_(self):
#         return self.a - self.b
    
# class TestMyClass(unittest.TestCase):
#     @classmethod
#     def setUp(self):
#         self.a = [1,2,3]
#         self.b = [4,5,6]
#         self.zipped = list(itertools.product(self.a, self.b))

#     # def loop_combinations(cls, *attributes):
#     #     def decorator(func):
#     #         def wrapper(*args, **kwargs):
#     #             for attribute_values in zip(*attributes):
#     #                 func(*attribute_values, *args, **kwargs)
#     #         return wrapper
#     #     return decorator
        
#     def test_add(self):
#         for args in self.zipped:
#             instance = MyClass(*args)
#             self.assertEqual(instance.add_(), sum(args))

#     # @loop_combinations([1,2,3], [2,3,4])
#     # def test_mult(self, a, b):
#     #     instance= MyClass(a,b)
#     #     result = instance.mult_()
#     #     self.assertEqual(result, a*b)








if __name__ == '__main__':

    unittest.main()
    