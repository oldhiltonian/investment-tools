import unittest
import requests
import pandas as pd
import numpy as np
import datetime as dt
from fmp_layer_1 import FinancialData, ManualAnalysis, Plots, Company
from pathlib import Path
import itertools
import random

key_path = Path().home()/'desktop'/'FinancialModellingPrep_API.txt'
with open(key_path) as file:
    api_key = file.read()

class TestFinancialData(unittest.TestCase):
    """
    Test suite for the FinancialData class.

    Attributes:
        tickers (list): A list of stock tickers to test.
        api_key (str): An API key for the Financial Modeling Prep API.
        data (list): A list of data types to test.
        period (list): A list of periods to test.
        limit (int): A limit on the number of records to fetch.
        zipped_args_tdp (list): A list of tuples representing all combinations
            of tickers, data, and period.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test suite.

        This method is called before any test methods are run.
        """
        # self.tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
        cls.tickers = ['AAPL']
        cls.api_key = api_key
        cls.data =    ['online', 'local']
        cls.period =  ['annual', 'quarter']
        cls.limit = 120
        cls.zipped_args_tdp = list(itertools.product(cls.tickers, cls.data, cls.period))
        cls.generic_instance = FinancialData('AAPL', cls.api_key, 'local', 'annual', 10)

    def test_assert_valid_user_input(self):
        """
        Test that the assert_valid_user_inputs method raises a ValueError when an
        invalid argument is passed to the FinancialData class constructor.

        This method creates a new FinancialData instance for each given ticker and test period
        and calls the assert_valid_user_inputs method for each instance.

        Args:
            self: An instance of the unittest.TestCase class.

        Returns:
            None.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            instance.assert_valid_user_inputs()
            instance.period = 1
            with self.assertRaises(AssertionError):
                instance.assert_valid_user_inputs()
        
#     def test_generate_request_url(self):
#         """
#         Test that the generate_request_url method returns the expected URL strings for the
#         balance sheet, income statement, cash flow statement, and reported key metrics
#         endpoints, and raises a ValueError for any other input.

#         This method creates a new FinancialData instance for each given ticker and test period
#         and generates the expected URL strings for each endpoint. It then checks that the
#         generate_request_url method returns the expected URL string and that a ValueError
#         is raised for invalid input.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             bs_str = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
#             is_str = f'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
#             cfs_str = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
#             metric_str = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={period}&limit={self.limit}&apikey={api_key}'
#             self.assertEqual(instance.generate_request_url('bs'), bs_str)
#             self.assertEqual(instance.generate_request_url('is'), is_str)
#             self.assertEqual(instance.generate_request_url('cfs'), cfs_str)
#             self.assertEqual(instance.generate_request_url('metrics'), metric_str)
#             with self.assertRaises(ValueError):
#                 instance.generate_request_url('')
#                 instance.generate_request_url(4)
#                 instance.generate_request_url('42')

#     def test_fetch_raw_data(self):
#         """
#         Test that the fetch_raw_data method returns a pd.DataFrame object when the 'local' argument
#         is passed to the FinancialData class constructor and a requests.Response object when the
#         'online' argument is passed, and raises a ValueError for any other input.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         calls the fetch_raw_data method for each endpoint. It checks that the returned object is of
#         the expected type and that a ValueError is raised for invalid input.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, api_key, data, period)
#             for string in ['bs', 'is', 'cfs', 'metrics']:
#                 raw_data = instance.fetch_raw_data(string)
#                 expected_type = requests.Response if data == 'online' else pd.DataFrame
#                 self.assertEqual(isinstance(raw_data, expected_type), True)
#                 with self.assertRaises(ValueError):
#                     instance.fetch_raw_data('')
#                     instance.fetch_raw_data(4)
#                     instance.fetch_raw_data(42)
        
#     def test_get_load_path(self):
#         """
#         Test that the get_load_path method returns the expected file path for the balance sheet,
#         income statement, cash flow statement, reported key metrics, and stock price data
#         for the given FinancialData instance.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         generates the expected file paths for each endpoint. It then checks that the get_load_path
#         method returns the expected file path.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             bs_str = (f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
#                             Company_Financial_Data\\{ticker}\\{period}\\balance_sheets.parquet''')
#             is_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
#                             Company_Financial_Data\\{ticker}\\{period}\\income_statements.parquet'''
#             cfs_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
#                         Company_Financial_Data\\{ticker}\\{period}\\cash_flow_statements.parquet'''
#             metric_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
#                         Company_Financial_Data\\{ticker}\\{period}\\reported_key_metrics.parquet'''
#             price_str = f'''C:\\Users\\John\\Desktop\\Git\\investment-tools\\investment_tools\\data\\
#                         Company_Financial_Data\\{ticker}\\{period}\\stock_price_data.parquet'''
#             bs_str = bs_str.replace('\n', '').replace('\t', '').replace(' ', '')
#             is_str = is_str.replace('\n', '').replace('\t', '').replace(' ', '')
#             cfs_str = cfs_str.replace('\n', '').replace('\t', '').replace(' ', '')
#             metric_str = metric_str.replace('\n', '').replace('\t', '').replace(' ', '')
#             price_str = price_str.replace('\n', '').replace('\t', '').replace(' ', '')

#             self.assertEqual(str(Path(bs_str)), str(instance.get_load_path('bs', ticker, period)))
#             self.assertEqual(str(Path(is_str)), str(instance.get_load_path('is', ticker, period)))
#             self.assertEqual(str(Path(cfs_str)), str(instance.get_load_path('cfs', ticker, period)))
#             self.assertEqual(str(Path(metric_str)), str(instance.get_load_path('metrics', ticker, period)))
#             self.assertEqual(str(Path(price_str)), str(instance.get_load_path('price', ticker, period)))

#     def test_get_frame_indecies(self):
#         """
#         Test that the get_frame_indecies method returns the correct index for the balance sheet,
#         income statement, cash flow statement, reported key metrics, and stock price data for the
#         given FinancialData instance.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         checks that the index of each financial statement's DataFrame is identical to the expected
#         index returned by get_frame_indecies.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             for item in [instance.balance_sheets, instance.income_statements,
#                               instance.cash_flow_statements, instance.reported_key_metrics,
#                               instance.stock_price_data]:
#                 expected = item.index
#                 self.assertEqual(instance.get_frame_indecies().equals(expected), True)

#     def test_set_frame_index(self):
#         """
#         Test that the set_frame_index method sets the index of the balance sheet,
#         income statement, cash flow statement, reported key metrics, and stock price data
#         for the given FinancialData instance to the expected index.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         sets the index of each financial statement's DataFrame to the expected index using the
#         set_frame_index method. The method then checks that the index of each financial statement's
#         DataFrame is identical to the expected index.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             length = len(instance.balance_sheets)
#             expected = pd.Index([str(i) for i in range(length)])
#             instance.frame_indecies = expected
#             for item in [instance.balance_sheets, instance.income_statements,
#                         instance.cash_flow_statements, instance.reported_key_metrics,
#                         instance.stock_price_data]:
#                 instance.set_frame_index(item)
#                 result = item.index
#                 self.assertEqual(expected.equals(result), True)

#    # '''Design a test for build_dataframe'''

#     def test_generate_index(self):
#         """
#         Test that the generate_index method returns the expected index for a given date and FinancialData instance.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         checks that the index generated by the generate_index method for each date in the test_dates list
#         is identical to the expected index.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         test_dates = ['1900-09-10', '1945-12-12', '2020-01-01', '2022-05-02']
#         for date in test_dates:
#             for ticker, data, period in self.zipped_args_tdp:
#                 instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#                 year, month, _ = [int(i) for i in date.split('-')]
#                 if instance.period == 'annual':
#                     expected = f"{instance.ticker}-FY-{year}"
#                 else:
#                     if month in (1,2,3):
#                         quarter = 1
#                     elif month in (4,5,6):
#                         quarter = 2
#                     elif month in (7,8,9):
#                         quarter = 3
#                     elif month in (10, 11, 12):
#                         quarter = 4
#                     expected = f"{instance.ticker}-Q{quarter}-{year}"
#                 result = instance.generate_index(date)
#                 self.assertEqual(result, expected)

#     def test_generate_date(self):
#         """
#         Test that the generate_date method returns the expected datetime.date object for a given date string and
#         FinancialData instance.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         checks that the datetime.date object generated by the generate_date method for each date in the
#         test_dates list is identical to the expected datetime.date object.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         test_dates = ['1900-09-10', '1945-12-12', '2020-01-01', '2022-05-02']
#         for date in test_dates:
#             for ticker, data, period in self.zipped_args_tdp:
#                 instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#                 year, month, day = [int(i) for i in date.split()[0].split('-')]
#                 expected = dt.date(year, month, day)
#                 result = instance.generate_date(date)
#                 self.assertEqual(expected, result)

#     def test_check_for_matching_indecies(self):
#         """
#         Test that the check_for_matching_indecies method returns True when all financial statements have the same index.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         sets the index of each financial statement's DataFrame to a new DataFrame containing a 'date' column
#         with the same length as the balance_sheets DataFrame. It then checks that the check_for_matching_indecies
#         method returns True. The method then sets the index of the balance_sheets DataFrame to an empty DataFrame
#         and checks that the check_for_matching_indecies method returns False.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             new_df = pd.DataFrame({'date': [str(i) for i in range(30)]})
#             instance.balance_sheets = new_df.copy()
#             instance.income_statements = new_df.copy()
#             instance.cash_flow_statements = new_df.copy()
#             instance.reported_key_metrics = new_df.copy()
#             expected = True
#             result = instance.check_for_matching_indecies()
#             self.assertEqual(expected, result)
#             instance.balance_sheets = pd.DataFrame({'date': []})
#             expected = False
#             result = instance.check_for_matching_indecies()
#             self.assertEqual(expected, result)
            
#     def test_get_common_df_indecies(self):
#         """
#         Test that the get_common_df_indicies method returns the expected index for a given FinancialData instance.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         sets the balance sheet, income statement, cash flow statement, and reported key metrics DataFrames
#         to new DataFrames with different indices. It then checks that the get_common_df_indicies method
#         returns the expected common index.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             instance.balance_sheets = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
#             instance.income_statements = pd.DataFrame({'B': [4, 5, 6]}, index=[1, 2, 0])
#             instance.cash_flow_statements = pd.DataFrame({'C': [6, 7, 8]}, index=[2, 3, 0])
#             instance.reported_key_metrics = pd.DataFrame({'D': [8, 9, 0]}, index=[3, 2, 8])
#             expected = pd.Index([2])
#             result = instance.get_common_df_indicies()
#             self.assertEqual(expected, result)
        
#     def test_filter_for_common_indecies(self):
#         """
#         Test that the filter_for_common_indecies method filters all financial statements
#         for the given FinancialData instance to only include rows with common indices.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         sets the balance sheet, income statement, cash flow statement, and reported key metrics DataFrames
#         to new DataFrames with different indices. It then filters each DataFrame using the
#         filter_for_common_indecies method and checks that the get_common_df_indicies method
#         returns the expected common index.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             instance.balance_sheets = pd.DataFrame({'A': [1, 2, 3]}, index=[0, 1, 2])
#             instance.income_statements = pd.DataFrame({'B': [4, 5, 6]}, index=[1, 2, 0])
#             instance.cash_flow_statements = pd.DataFrame({'C': [6, 7, 8]}, index=[2, 3, 0])
#             instance.reported_key_metrics = pd.DataFrame({'D': [8, 9, 0]}, index=[3, 2, 8])
#             instance.filter_for_common_indecies(pd.Index([2]))
#             result = instance.get_common_df_indicies()
#             self.assertEqual(pd.Index([2]), result)


#     def test_assert_identical_indecies(self):
#         """
#         Test that the assert_identical_indecies method raises an AssertionError when
#         the indices of the balance sheet, income statement, cash flow statement,
#         reported key metrics, and stock price data for the given FinancialData instance
#         are not identical.

#         This method creates a new FinancialData instance for each given ticker and test period and
#         sets the index of the balance_sheets DataFrame to a new index. It then calls the
#         assert_identical_indecies method and checks that an AssertionError is raised. The method
#         then sets the index of the balance_sheets DataFrame back to its original index.

#         Args:
#             self: An instance of the unittest.TestCase class.

#         Returns:
#             None.
#         """
#         for ticker, data, period in self.zipped_args_tdp:
#             instance = FinancialData(ticker, self.api_key, data, period, self.limit)
#             new_index = pd.Index([str(i) for i in range(len(instance.balance_sheets))])
#             instance.assert_identical_indecies()
#             instance.balance_sheets.index = new_index
#             with self.assertRaises(AssertionError):
#                 instance.assert_identical_indecies()

    # def test_assert_required_length(self):
    #     for ticker, data, period in self.zipped_args_tdp:
    #         instance = FinancialData(ticker, self.api_key, data, period, self.limit)
    #         lengths = [4, 5, 6, 500, 1000] if period == 'quarter' else [2, 3, 7, 12, 1000]
    #         for length in lengths:
    #             instance.assert_required_length(range(length))
    #         fail = range(3) if period == 'quarter' else range(1)
    #         with self.assertRaises(AssertionError):
    #             instance.assert_required_length(fail)
            

    # def test_assert_valid_server_response(self):
    #     for ticker, _, period in self.zipped_args_tdp:
    #         instance = FinancialData(ticker, self.api_key, 'online', period, self.limit)
    #         response = instance.fetch_raw_data('bs')
    #         instance.assert_valid_server_response(response)

    def test_assert_server_response_not_empty(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = FinancialData(ticker, self.api_key, data, period, self.limit)
            response = requests.Response()
            response._content = b'{"key": "value"}'
            response.status_code = 200
            result = instance.assert_server_response_not_empty(response)
            self.assertIsNone(result)






if __name__ == '__main__':

    unittest.main()
    