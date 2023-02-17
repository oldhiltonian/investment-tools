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


'''Thus far I only have tests for the FinancialData class'''
# class TestFinancialData(unittest.TestCase):

#     # def __init__(self, ticker, data, period, api_key, limit=120):
#         # self.api_key = api_key
#         # self.ticker = ticker
#         # self.data = data
#         # self.period = period
#         # self.limit = limit
#         # self.data = FinancialData(self.ticker, self.api_key, self.data, self.period, self.limit)
   
#     def setUp(self):
#         self.api_key = api_key
#         self.ticker = 'AAPL'
#         self.data = 'online'
#         self.period = 'annual'
#         self.limit = 120
#         self.data = FinancialData(self.ticker, self.api_key, self.data, self.period, self.limit)
        
#     def test_class_attributes(self):
#         self.assertEqual(self.ticker, ticker)
#         self.assertEqual(self.period, period)
#         self.assertEqual(self.data.days_in_period, 365 if self.data.period == 'annual' else 90)

#     def test_length_of_statements(self):
#         self.assertNotEqual(len(self.data.balance_sheets), 0)
#         self.assertNotEqual(len(self.data.income_statements), 0)
#         self.assertNotEqual(len(self.data.cash_flow_statements), 0)
#         self.assertEqual(len(self.data.balance_sheets), len(self.data.income_statements))
#         self.assertEqual(len(self.data.balance_sheets), len(self.data.cash_flow_statements))

#     def test_matching_index(self):
#         self.assertEqual(self.data.balance_sheets.index.tolist(), self.data.income_statements.index.tolist())
#         self.assertEqual(self.data.balance_sheets.index.tolist(), self.data.cash_flow_statements.index.tolist())
    

#     def test_fetch_financial_statements(self):
#         balance_sheets, income_statements, cash_flow_statements = self.data.fetch_financial_statements(self.ticker, self.api_key, self.period, self.limit)
#         self.assertIsInstance(balance_sheets, list)
#         self.assertIsInstance(income_statements, list)
#         self.assertIsInstance(cash_flow_statements, list)
#         self.assertIsInstance(balance_sheets[0], dict)
#         self.assertIsInstance(income_statements[0], dict)
#         self.assertIsInstance(cash_flow_statements[0], dict)

        


#     def test_build_dataframe(self):
#         statements = [{'date': '2021-01-01', 'key1': 1, 'key2': 2}, {'date': '2021-02-01', 'key1': 3, 'key2': 4}]
#         df = self.data.build_dataframe(statements)
#         self.assertIsInstance(df, pd.DataFrame)
#         self.assertEqual(df.shape[0], 2)
#         self.assertEqual(df.shape[1], 3)
#         self.assertEqual(df.columns.tolist(), ['date', 'key1', 'key2'])
#         self.assertIsInstance(df['date'][0], dt.date)


#     def test_generate_index(self):
#         if self.period == 'annual':
#             date = "2021-01-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-FY-2021")
#             date = "2021-04-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-FY-2021")
#             date = "1985-12-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-FY-1985")

#         elif self.period == 'quarter':
#             date = "2021-01-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-Q1-2021")
#             date = "2021-04-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-Q2-2021")
#             date = "2021-08-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-Q3-2021")
#             date = "1985-12-01"
#             index = self.data.generate_index(date)
#             self.assertEqual(index, f"{self.ticker}-Q4-1985")

#     def test_generate_date(self):
#         '''test to ensure an exception is thrown when the input date is not in the correct format'''
#         date_strings = ["2021-01-01", "1985-10-15", "2000-11-11", "1999-08-20"]
#         for date_str in date_strings:
#             year, month, day = [int(i) for i in date_str.split('-')]
#             date = self.data.generate_date(date_str)
#             self.assertIsInstance(date, dt.date)
#             self.assertEqual(date, dt.date(year, month, day))
        




class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.api_key = api_key
        self.ticker = 'AAPL'
        self.data = 'online'
        self.period = 'annual'
        self.limit = 120
        self.data = FinancialData(self.ticker, self.api_key, self.data, self.period, self.limit)
        self.analysis = ManualAnalysis(self.data)

    def test_cross_check(self):
        return_dfs = self.analysis.cross_check_statement_calculations()
        calculated, reported, metric_errors, ratio_errors = return_dfs
        len_calculated = len(calculated)
        self.assertGreater(len_calculated, 0)
        comparison_sum = sum(calculated.index == self.data.frame_indecies)
        self.assertEqual(comparison_sum, len_calculated)
        comparison_sum = sum(calculated.index == reported.index)
        self.assertEqual(comparison_sum, len_calculated)
        comparison_sum = sum(calculated.index == metric_errors.index)
        self.assertEqual(comparison_sum, len_calculated)
        comparison_sum = sum(calculated.index == ratio_errors.index)
        self.assertEqual(comparison_sum, len_calculated)
        self.assertEqual(set(calculated.keys()), set(reported.keys()))
        self.assertEqual(set(calculated.keys()), set(metric_errors.keys()))

    def test_analyse(self):
        df = self.analysis.analyse()
        len_df = len(df)
        # self.assertEqual(df.index, self.data.frame_indecies)
        self.assertGreater(len_df, 0)
        comparison_sum = sum(df['PE_low'] <= df['PE_high'])
        self.assertEqual(comparison_sum, len_df)
        comparison_sum = sum(df['PE_low'] <= df['PE_avg_close'])
        self.assertequal(comparison_sum, len_df)
        comparison_sum = sum(df['dividendYield_low'] <= df['dividendYield_high'])
        self.assertequal(comparison_sum, len_df)
        comparison_sum = sum(df['dividendYield_low'] <= df['dividendYield_avg_close'])
        self.assertequal(comparison_sum, len_df)
        comparison_sum = sum(df['netProfitMargin'] <= df['pretaxProfitMargin'])
        self.assertequal(comparison_sum, len_df)
        comparison_sum = sum(df['cashRatio'] <= df['quickRatio'])
        self.assertequal(comparison_sum, len_df)
        comparison_sum = sum(df['quickRatio'] <= df['currentRatio'])
        self.assertequal(comparison_sum, len_df)


        




if __name__ == '__main__':

    # tickers = ['AAPL', 'BOC', 'NVDA', 'JXN']
    # data_sources = ['online', 'local']
    # periods = ['annual', 'quarterly']

    ticker = 'AAPL'
    data = 'online'
    period = 'annual'
    unittest.main()
    