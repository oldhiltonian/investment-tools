import unittest
import requests
import pandas as pd
import datetime as dt
from FMP import FinancialData
from pathlib import Path

key_path = Path().home()/'desktop'/'FinancialModellingPrep_API.txt'
with open(key_path) as file:
    api_key = file.read()

'''Thus far I only have tests for the FinancialData class'''
class TestFinancialData(unittest.TestCase):
    def setUp(self):
        self.api_key = api_key
        self.ticker = 'AAPL'
        self.data = 'online'
        self.period = 'annual'
        self.limit = 120
        self.data = FinancialData(self.ticker, self.api_key, self.data, self.period, self.limit)
    
    def add_test(self, ticker, data, period, limit):
        self.ticker.append(ticker)
        self.data.append(data)
        self.period.append(period)
        self.limit.append(limit)

    def test_length_of_statements(self):
        '''returned frames cannot have a length of zero
            and all must have equal lengths'''
        self.assertNotEqual(len(self.data.balance_sheets), 0)
        self.assertNotEqual(len(self.data.income_statements), 0)
        self.assertNotEqual(len(self.data.cash_flow_statements), 0)
        self.assertEqual(len(self.data.balance_sheets), len(self.data.income_statements))
        self.assertEqual(len(self.data.balance_sheets), len(self.data.cash_flow_statements))

    def test_matching_index(self):
        '''Frames must have the same index to facilitate correct calculations of financial ratios'''
        self.assertEqual(self.data.balance_sheets.index.tolist(), self.data.income_statements.index.tolist())
        self.assertEqual(self.data.balance_sheets.index.tolist(), self.data.cash_flow_statements.index.tolist())
    

    def test_fetch_financial_statements(self):
        '''method must return a tuple of json objects
            method must throw an exception if the API returns an error'''
        balance_sheets, income_statements, cash_flow_statements = self.data.fetch_financial_statements(self.ticker, self.api_key, self.period, self.limit)
        self.assertIsInstance(balance_sheets, list)
        self.assertIsInstance(income_statements, list)
        self.assertIsInstance(cash_flow_statements, list)
        self.assertIsInstance(balance_sheets[0], dict)
        self.assertIsInstance(income_statements[0], dict)
        self.assertIsInstance(cash_flow_statements[0], dict)
        pass


    def test_build_dataframe(self):
        statements = [{'date': '2021-01-01', 'key1': 1, 'key2': 2}, {'date': '2021-02-01', 'key1': 3, 'key2': 4}]
        df = self.data.build_dataframe(statements)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 2)
        self.assertEqual(df.shape[1], 4)
        self.assertEqual(df.columns.tolist(), ['date', 'key1', 'key2', 'index'])
        self.assertIsInstance(df['date'][0], dt.date)


    def test_generate_index(self):
        '''method must return a string of a specific format based on annual or quarter
            missing checking self.period for annual or quarter'''
        date = "2021-01-01"
        index = self.data.generate_index(date)
        self.assertEqual(index, f"{self.ticker}-FY-2021")
        date = "2021-04-01"
        index = self.data.generate_index(date)
        self.assertEqual(index, f"{self.ticker}-FY-2021")

    def test_generate_date(self):
        '''test to ensure that a date object is returned 
            test to ensure an exception is thrown when the input date is not in the correct format'''
        date_str = "2021-01-01"
        date = self.data.generate_date(date_str)
        self.assertIsInstance(date, dt.date)
        self.assertEqual(date, dt.date(2021, 1, 1))

    def test_filter_for_common_indices(self):
        '''Test that the method correctly filters the three financial statement DataFrames to only contain rows with matching indices'''
        pass


if __name__ == '__main__':


    ticker = 'AAPL'
    data = 'online'
    period = 'annual'
    
    unittest.main()