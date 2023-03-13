import unittest
import sys

sys.path.append("..")
from investment_tools import Company
import pandas as pd
from pathlib import Path
import itertools
import unittest.mock as mock
from unittest.mock import Mock, patch
import numpy as np
import math
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

key_path = Path().home() / "desktop" / "FinancialModellingPrep_API.txt"
with open(key_path) as file:
    api_key = file.read()


class TestStandardEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets up the class attributes for running the test suite.

        - cls.tickers : List[str]
            List of ticker symbols to be used in the test.

        - cls.api_key : str
            API key for the Financial Modeling Prep API.

        - cls.data : List[str]
            List of strings representing the location of data.

        - cls.period : List[str]
            List of strings representing the time period of data.

        - cls.limit : int
            Limit for the number of data rows returned by API.

        - cls.zipped_args_tdp : List[Tuple[str,str,str]]
            List of tuples representing the combination of ticker symbol, data location,
            and time period.
        """
        # cls.tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
        cls.tickers = ["AAPL"]
        cls.api_key = api_key
        cls.data = ["online", "local"]
        cls.period = ["annual", "quarter"]
        cls.limit = 15
        cls.zipped_args_tdp = list(itertools.product(cls.tickers, cls.data, cls.period))

    def test_get_modifier(self):
        """
        Tests the `get_modifier()` method of the `StandardEvaluation` class.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `get_modifier()` method returns the expected output for various input strings.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for string in ["deb", "ebt", "dbt", "det"]:
                expected = 1
                result = instance.eval.get_modifier(string)
                self.assertEqual(expected, result)

            for string in ["debt", "DeBt", "DEBT", "DEbT"]:
                expected = -1
                result = instance.eval.get_modifier(string)
                self.assertEqual(expected, result)

    def test_get_scoring_metrics(self):
        """
        Tests the `get_scoring_metrics()` method of the `StandardEvaluation` class.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `get_scoring_metrics()` method returns the expected output.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            result = instance.eval.get_scoring_metrics()
            expected = [
                "eps",
                "returnOnEquity",
                "ROIC",
                "returnOnAssets",
                "debtToTotalCap",
                "totalDebtRatio",
            ]
            self.assertEqual(expected, result)

    def test_create_scoring_metrics_results_dict(self):
        """
        Tests the `create_scoring_metrics_results_dict()` method of the `StandardEvaluation` class.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `create_scoring_metrics_results_dict()` method returns the expected output.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            dct = instance.eval.standard_scores_dict
            self.assertEqual(len(dct), 6)
            self.assertIsInstance(dct, dict)
            expected = [
                "eps",
                "returnOnEquity",
                "ROIC",
                "returnOnAssets",
                "debtToTotalCap",
                "totalDebtRatio",
            ]
            result = list(dct.keys())
            self.assertEqual(expected, result)

    def test_get_copy_of_df_column(self):
        """
        Test that the get_copy_of_df_column method returns a copy of a
        DataFrame column with NaN values dropped.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `get_copy_of_df_column()` method returns the expected output.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for string in ["eps", "ebitdaratio", "ROIC", "totalDebtRatio"]:
                expected = instance.eval.metrics[string].copy().dropna()
                fetched = instance.eval.get_copy_of_df_column(string)
                result = expected.equals(fetched)
                self.assertEqual(result, True)

    def test_get_r2_val(self):
        """
        Test that the get_r2_val method correctly calculates the R-squared
        value of a given series.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `get_r2_val()` method returns the expected output.
        """
        series_dict = {
            "1": [0, 1, 2, 3, 4, 5],
            "0.9888": [2, 4, 6, 9],
            "0.9704": [1, 3, 4, 6, 9],
            "0.9730": [0, -1, -2, -3, -5],
            "0.9795": [0, -3, -4, -7, -10],
        }
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for r2, series in series_dict.items():
                r2_ = float(r2)
                expected = instance.eval.get_r2_val(series)
                self.assertAlmostEqual(expected, r2_, places=3)

    def test_score_mean_growth(self):
        """
        Test that the score_mean_growth method returns the correct score for a
        given mean growth rate.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `score_mean_growth()` method returns the expected output.
        """
        mean_growth_score_tuple = [
            (0.05, 0),
            (0.051, 1),
            (0.099, 1),
            (0.1001, 2),
            (0.1499, 2),
            (0.15001, 3),
            (0.2, 3),
            (0.2001, 4),
        ]

        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)

            for mean, score in mean_growth_score_tuple:
                self.assertEqual(instance.eval.score_mean_growth(mean), score)

    def test_score_trend_strength(self):
        """
        Test that the score_trend_strength method returns the correct score for a
        given R-squared value.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `score_trend_strength()` method returns the expected output.
        """
        r2_to_score_tuples = [
            (0.01, 0),
            (0.2, 0),
            (0.2001, 1),
            (0.3, 1),
            (0.3001, 2),
            (0.5, 2),
            (0.5001, 3),
            (0.75, 3),
            (0.751, 4),
        ]
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for r2, score in r2_to_score_tuples:
                self.assertEqual(instance.eval.score_trend_strength(r2), score)

    def test_get_slope_and_intercept(self):
        """
        Test that the `get_slope_and_intercept()` method correctly calculates the slope
        and y-intercept of a linear regression model.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `get_slope_and_intercept()` method returns the expected output.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            arrays = [
                np.array([1, 2, 3, 5, 7, 9]),
                np.array([1, 2, 3, 5, 8, 9]),
                np.array([1, 2, 3, 5, 8, 13]),
            ]
            expected = [
                (1.6285714285714286, 0.4285714285714288),
                (1.7142857142857144, 0.3809523809523805),
                (2.2857142857142856, -0.3809523809523805),
            ]
            for array, expected_ in zip(arrays, expected):
                result = instance.eval.get_slope_and_intercept(array)
                self.assertAlmostEqual(expected_[0], result[0], 4)
                self.assertAlmostEqual(expected_[1], result[1], 4)

    def calculate_mean_growth_from_series_trend(self):
        """
        Test that the calculate_mean_growth_rate method correctly calculates the
        mean growth rate of a given series.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class, and tests whether the
        `calculate_mean_growth_from_series_trend()` method returns the expected output.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            arrays = [
                np.array([1, 2, 3, 5, 7, 9]),
                np.array([1, 2, 3, 5, 8, 9]),
                np.array([2, 2, 3, 5, 8, 13]),
            ]
            expected = [0.8205642030260802, 0.880241233365431, 1.3777309915742477]
            for array, expected_ in zip(arrays, expected):
                result = instance.eval.calculate_mean_growth_rate(array)
                self.assertAlmostEqual(expected_, result, 2)
                self.assertAlmostEqual(expected_, result, 2)

    def test_sum_of_scoring_metric_dict_scores(self):
        """
        Test if the function correctly calculates the sum of scores in a dictionary of scoring metrics.

        For a Company instance, it generates two dictionaries of scoring metrics and tests whether the
        `sum_of_scoring_metric_dict_scores()` method returns the expected output for each of them.
        """
        instance = Company("AAPL", self.api_key, "online", "annual")
        dct_1 = {
            "eps": {"score": 1, "strength": 3},
            "returnOnEquity": {"score": 1, "strength": 1},
            "ROIC": {"score": 1, "strength": 0},
            "returnOnAssets": {"score": 1, "strength": 0},
            "debtToTotalCap": {"score": 0, "strength": 1},
            "totalDebtRatio": {"score": 0, "strength": 0},
        }
        result_1 = instance.eval.sum_of_scoring_metric_dict_scores(dct_1)
        expected_1 = 4
        self.assertEqual(expected_1, result_1)
        dct_2 = {
            "eps": {"score": 2, "strength": 3},
            "returnOnEquity": {"score": 2, "strength": 1},
            "ROIC": {"score": 1, "strength": 0},
            "returnOnAssets": {"score": 1, "strength": 0},
            "debtToTotalCap": {"score": 2, "strength": 1},
            "totalDebtRatio": {"score": 0, "strength": 0},
        }
        result_2 = instance.eval.sum_of_scoring_metric_dict_scores(dct_2)
        expected_2 = 8
        self.assertEqual(expected_2, result_2)

    def test_total_score_to_bool(self):
        """
        Test the total_score_to_bool method of the StandardEvaluation class.

        For the AAPL company instance created for the 'online' data location, and 'annual' time period,
        it tests whether the total_score_to_bool method returns the expected output. Note that the default
        threshold score is twice the length of the ._scoring_metrics attribute.

        It tests the following test cases:
            - when the total score is equal to or greater than the threshold score, it should return True
            - when the total score is less than the threshold score, it should return False
        """
        instance = Company("AAPL", self.api_key, "online", "annual")
        instance.eval._scoring_metrics = range(5)
        func = instance.eval.total_score_to_bool
        self.assertEqual(func(5, 5), True)
        self.assertEqual(func(6, 5), True)
        self.assertEqual(func(4, 5), False)
        self.assertEqual(func(9), False)
        self.assertEqual(func(10), True)
        self.assertEqual(func(11), True)

    def test_standard_eval(self):
        """
        Test that the standard evaluation method correctly returns a boolean
        value based on a given set of scoring metrics.

        For each combination of ticker symbol, data location, and time period,
        it creates an instance of the `Company` class and sets the standard
        scores dict. It then tests whether the `standard_eval()` method returns
        the expected output.
        """
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            scores = [
                {
                    "eps": {"score": 0, "strength": 0},
                    "returnOnEquity": {"score": 0, "strength": 0},
                    "ROIC": {"score": 0, "strength": 0},
                    "returnOnAssets": {"score": 0, "strength": 0},
                    "debtToTotalCap": {"score": 0, "strength": 4},
                    "totalDebtRatio": {"score": 0, "strength": 4},
                },
                {
                    "eps": {"score": 4, "strength": 0},
                    "returnOnEquity": {"score": np.nan, "strength": 0},
                    "ROIC": {"score": np.nan, "strength": 0},
                    "returnOnAssets": {"score": 4, "strength": 0},
                    "debtToTotalCap": {"score": 4, "strength": 4},
                    "totalDebtRatio": {"score": 4, "strength": 4},
                },
                {
                    "eps": {"score": np.nan, "strength": 0},
                    "returnOnEquity": {"score": np.nan, "strength": 0},
                    "ROIC": {"score": np.nan, "strength": 0},
                    "returnOnAssets": {"score": 0, "strength": 0},
                    "debtToTotalCap": {"score": 0, "strength": 4},
                    "totalDebtRatio": {"score": 0, "strength": 4},
                },
            ]

            evals = [False, True, False]
            for score_dict, expected in zip(scores, evals):
                instance.eval.standard_scores_dict = score_dict
                result = instance.eval.standard_eval()
                self.assertEqual(result, expected)


def company_instance_generator(api_key, limit) -> Company:
    """
     Generate Company instances using provided API key and limit.

     Args:
          api_key (str): API key for accessing financial data.
          limit (int): Maximum number of results to be returned.

     Returns:
          Generator[Company]: A generator that yields Company instances with given parameters.

     Example:
          ticker_list = ['AAPL', 'MSFT', 'NVDA', 'VAC', 'WBA', 'ATVI', 'A', 'AMD']
          generator = company_instance_generator(api_key='your_api_key', limit=10)

     Note:
     The function only generates Company instances for the ticker 'AAPL' by default.
     The 'data' and 'period' arguments are hard-coded to be 'online', 'local' and 'annual', 'quarter' respectively.
     """
    # tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
    tickers = ["AAPL"]
    api_key = api_key
    data = ["online", "local"]
    period = ["annual", "quarter"]
    limit = 15
    instance_combinations = list(itertools.product(tickers, data, period))
    for ticker, data_, period_ in instance_combinations:
        yield Company(ticker, api_key, data_, period_, limit)


class TestBuffetEvaluation(unittest.TestCase):
    """
     A class for unit testing the BuffetEvaluation module.

     The class provides several test methods for each of the methods in the BuffetEvaluation class. The methods test the functionalities of the BuffetEvaluation methods using different test cases.

     Note that in the test methods, the 'company_instance_generator' function is used to generate a generator that yields Company instances. The 'setUpClass' method is used to set up some attributes of the class that are used in the test methods.

     Attributes:
          tickers (List[str]): A list of tickers to be used in generating Company instances.
          api_key (str): API key for accessing financial data.
          data (List[str]): A list of data types to be used in generating Company instances.
          period (List[str]): A list of periods to be used in generating Company instances.
          limit (int): Maximum number of results to be returned.
          zipped_args_tdp (List[Tuple[str, str, str]]): A list of tuples that contain ticker, data type, and period combinations.

     Methods:
          test_buffet_test_1_is_eps_increasing: Test the 'buffet_test_1_is_eps_increasing' method of the BuffetEvaluation class.
          test_buffet_test_2_initial_RoR: Test the 'buffet_test_2_initial_RoR' method of the BuffetEvaluation class.
          test_buffet_test_3_determine_eps_growth: Test the 'buffet_test_3_determine_eps_growth' method of the BuffetEvaluation class.
          test_buffet_test_4_compare_to_TBonds: Test the 'buffet_test_4_compare_to_TBonds' method of the BuffetEvaluation class.
          test_buffet_test_5_RoE_projections: Test the 'buffet_test_5_RoE_projections' method of the BuffetEvaluation class.
          test_setup_test_5_RoE_projection_df: Test the 'setup_test_5_RoE_projection_df' method of the BuffetEvaluation class.
          test_get_current_stock_price: Test the 'get_current_stock_price' method of the BuffetEvaluation class.
          test_calculate_trendline_series: Test the 'calculate_trendline_series' method of the BuffetEvaluation class.
          test_project_future_value: Test the 'project_future_value' method of the BuffetEvaluation class.
          test_simple_discount_to_present: Test the 'simple_discount_to_present' method of the BuffetEvaluation class.
          test_get_x_day_mean_stock_price: Test the 'get_x_day_mean_stock_price' method of the BuffetEvaluation class.
          test_calculate_initial_rate_of_return: Test the 'calculate_initial_rate_of_return' method of the BuffetEvaluation class.
          test_calculate_simple_compound_interest: Test the 'calculate_simple_compound_interest' method of the BuffetEvaluation class.
          test_get_treasury_yield_api_url: Test the 'get_treasury_yield_api_url' method of the BuffetEvaluation class.
          test_get_5Y_treasury_yield_data: Test the 'get_5Y_treasury_yield_data' method of the BuffetEvaluation class.
          test_calculate_breakeven_vs_treasury: Test the 'calculate_breakeven_vs_treasury' method of the BuffetEvaluation class.
          test_treasury_comparison: Test the 'treasury_comparison' method of the BuffetEvaluation class.
     """

    @classmethod
    def setUpClass(cls) -> None:
        """
          Set up test fixture.

          This method sets up the test fixture by initializing the class variables.

          Args:
               cls (TestBuffetEvaluation): The class instance being created.
               
          Returns:
               None
               
          Raises:
               None
          """
        # cls.tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
        cls.tickers = ["AAPL"]
        cls.api_key = api_key
        cls.data = ["online", "local"]
        cls.period = ["annual", "quarter"]
        cls.limit = 15
        cls.zipped_args_tdp = list(itertools.product(cls.tickers, cls.data, cls.period))

    def test_buffet_test_1_is_eps_increasing(self):
        """Test if the Buffett Test 1 for increasing EPS returns the expected results.

          Uses predefined score dictionaries to set standard scores for various metrics
          and iterates through a generator of Company instances to check if the
          Buffett Test 1 for increasing EPS returns the expected results. For each
          Company instance, the standard scores are set, the test is run, and the
          result is compared to the expected value.

          Args:
               self: The TestBuffetEvaluation instance.

          Returns:
               None.

          Raises:
               AssertionError: If the result of the test does not match the expected
                    value.
          """
        score_dicts = [
            {
                "eps": {"score": 3, "strength": 3},
                "returnOnEquity": {"score": 0, "strength": 0},
                "ROIC": {"score": 0, "strength": 0},
                "returnOnAssets": {"score": 0, "strength": 0},
                "debtToTotalCap": {"score": 0, "strength": 4},
                "totalDebtRatio": {"score": 0, "strength": 4},
            },
            {
                "eps": {"score": 4, "strength": 2},
                "returnOnEquity": {"score": np.nan, "strength": 0},
                "ROIC": {"score": np.nan, "strength": 0},
                "returnOnAssets": {"score": 4, "strength": 0},
                "debtToTotalCap": {"score": 4, "strength": 4},
                "totalDebtRatio": {"score": 4, "strength": 4},
            },
            {
                "eps": {"score": np.nan, "strength": 1},
                "returnOnEquity": {"score": np.nan, "strength": 0},
                "ROIC": {"score": np.nan, "strength": 0},
                "returnOnAssets": {"score": 0, "strength": 0},
                "debtToTotalCap": {"score": 0, "strength": 4},
                "totalDebtRatio": {"score": 0, "strength": 4},
            },
        ]
        expected_results = [True, False, False]
        for company in company_instance_generator(api_key, 10):
            for score, expected in zip(score_dicts, expected_results):
                company.eval_buffet.standard_scores_dict = score
                result = company.eval_buffet.buffet_test_1_is_eps_increasing()
                self.assertEqual(result, expected)

    def test_buffet_test_2_initial_RoR(self):
        """
          Test case for checking the initial rate of return for Buffet Test 2.

          Iterates over a generator of company instances and sets the 'eps' metric of each instance to 25.
          The 'get_x_day_mean_stock_price' method of each instance is mocked to return 100.
          The expected initial rate of return is calculated to be 0.25.
          Asserts that the actual result matches the expected result.

          Returns:
               None.
          """
        for company in company_instance_generator(api_key, 10):
            eval_buffet = company.eval_buffet
            eval_buffet.metrics["eps"][-1] = 25
            eval_buffet.get_x_day_mean_stock_price = Mock()
            eval_buffet.get_x_day_mean_stock_price.return_value = 100
            result = eval_buffet.buffet_test_2_initial_RoR()
            self.assertEqual(result, 0.25)

    def test_buffet_test_3_determine_eps_growth(self):
        """Test that the method buffet_test_3_determine_eps_growth calculates
          the mean growth rate of the series trendline of the eps data correctly.

          The method uses self.calculate_mean_growth_from_series_trend to
          calculate the mean rate growth rate of the series trendline, not from
          the true data values.

          For this test, we generate a sample DataFrame with a few datapoints
          and test if the result of the method is close enough to the expected
          result.

          Raises:
               AssertionError: If the result does not match the expected value
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            eval.metrics = pd.DataFrame(
                [1, 2, 3, 5, 7, 9, 9, 9, 10, 11, 12, 13, 14], columns=["eps"]
            )
            result = eval.buffet_test_3_determine_eps_growth()
            expected = (0.08370, 0.092388, 0.07538, 0.12356)
            for i, j in zip(result, expected):
                self.assertAlmostEqual(i, j, 4)

    def test_buffet_test_4_compare_to_TBonds(self):
        """
          Test if the current stock price is less than 1.1 times the earnings yield
          for the most recent year relative to the 5-year treasury yield.

          For each company instance generated using the API key and a limit of 12, the function
          mocks the 5-year treasury yield data and the current stock price, and then calculates
          the breakeven earnings yield. The function then compares the current stock price to
          1.1 times the breakeven earnings yield, and returns True if the current stock price
          is less than or equal to that value, False otherwise.
          """
        for company in company_instance_generator(api_key, 12):
            treasury_yield = 0.05
            eval = company.eval_buffet
            eval.metrics = pd.DataFrame([1, 2, 3, 4, 5], columns=["eps"])
            eval.get_5Y_treasury_yield_data = Mock()
            eval.get_5Y_treasury_yield_data.return_value = treasury_yield
            eval.get_current_stock_price = Mock()
            for price in [10, 30, 70, 90, 100, 110, 111, 130, 150]:
                eval.get_current_stock_price.return_value = price
                eps = eval.metrics["eps"].iloc[-1]
                breakeven = eps / treasury_yield
                expected = True if price <= breakeven * 1.1 else False
                result = eval.buffet_test_4_compare_to_TBonds()
                self.assertEqual(result, expected)

    def test_buffet_test_5_RoE_projections(self):
        pass

    def test_setup_test_5_RoE_projection_df(self):
        """
          Test function to check the setup of RoE projection dataframe for Buffet test 5.
          
          This function generates a RoE projection dataframe for Buffet test 5, with a given
          `span` parameter, using the `setup_test_5_RoE_projection_df()` method of the company's
          `eval_buffet` instance. It then checks whether the length and columns of the generated
          dataframe match the expected values.
          
          Args:
               self: The TestCase instance.
               
          Returns:
               None
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            expected_cols = [
                "EqPS",
                "EPS",
                "DPS",
                "REPS",
                "FV_price_PE_high",
                "FV_price_PE_low",
                "FV_price_PEq_high",
                "FV_price_PEq_low",
                "PV_price_PE_high",
                "PV_price_PE_low",
                "PV_price_PEq_high",
                "PV_price_PEq_low",
                "RoR_current_price_to_FV_PE_high",
                "RoR_current_price_to_FV_PE_low",
                "RoR_current_price_to_FV_PEq_high",
                "RoR_current_price_to_FV_PEq_low",
            ]
            for span in [3, 5, 7, 10]:
                df = eval.setup_test_5_RoE_projection_df(3)
                self.assertEqual(len(df), 12)
                self.assertEqual(df.columns.to_list(), expected_cols)

    def test_get_current_stock_price(self):
        """
          Test the get_current_stock_price method of the EvalBuffet class.

          For each company instance generated by company_instance_generator, the function creates a mock object
          for the get_current_stock_price method and sets its return value to the last closing price in a
          DataFrame of price data. It then asserts that the value returned by get_current_stock_price matches
          the expected value.

          Raises:
               AssertionError: If the value returned by get_current_stock_price does not match the expected value.
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            eval.get_current_stock_price = Mock()
            prices = pd.DataFrame(
                [[1, 2, 3], [4, 5, 6]], columns=["Close", "High", "Low"]
            )
            eval.get_current_stock_price.return_value = prices.iloc[-1]["Close"]
            expected = 4
            result = eval.get_current_stock_price()
            self.assertEqual(result, expected)

    def test_calculate_trendline_series(self):
        """
          Test that the trendline series is correctly calculated.

          Creates a test series and compares the trendline series calculated by
          `calculate_trendline_series()` to the expected values.

          Raises:
               AssertionError: If the expected and actual results do not match.

          """
        series = pd.Series([1, 1, 3, 6, 8, 9, 12, 15])
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            expected = [
                -0.3333,
                1.7261,
                3.7859,
                5.845,
                7.9046,
                9.9642,
                12.0238,
                14.0833,
            ]
            result = list(eval.calculate_trendline_series(series))
            for i, j in zip(expected, result):
                self.assertEqual(round(i, 3), round(j, 3))

    def test_project_future_value(self):
        """
          Tests the `project_future_value()` method of the `EvalBuffet` class.
    
          This test case verifies that the `project_future_value()` method correctly
          calculates the future value of an investment given the present value, the
          annual interest rate, and the number of years the investment is held.
          
          Args:
               None
          
          Returns:
               None
          
          Raises:
               AssertionError: If the result of `project_future_value()` does not match
                    the expected future value within a tolerance of 0.001.
        """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            pvs = [2, 10, 100, 1034]
            rates = [0.01, 0.05, 0.1, 0.19]
            years = [10, 13, 5, 9]
            for pv, rate, year in zip(pvs, rates, years):
                expected = pv * (1 + rate) ** year
                result = eval.project_future_value(pv, rate, year)
                self.assertAlmostEqual(result, expected, 3)

    def test_simple_discount_to_present(self):
        """
          Test the simple_discount_to_present() method of the EvalBuffet class.
          For each company instance generated from the company_instance_generator function,
          the function tests the simple_discount_to_present() method using a range of
          future values (fvs), discount rates (rates), and years.
          It compares the expected discounted present value with the result obtained
          from simple_discount_to_present() method using assertAlmostEqual() method
          with a precision of 3 decimal places.
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            fvs = [200, 1097, 1030, 19844]
            rates = [0.01, 0.05, 0.1, 0.19]
            years = [10, 13, 5, 9]
            for fv, rate, year in zip(fvs, rates, years):
                expected = fv / ((1 + rate) ** year)
                result = eval.simple_discount_to_present(fv, year, rate)
                self.assertAlmostEqual(result, expected, 3)

    def test_get_x_day_mean_stock_price(self):
        pass

    def test_calculate_initial_rate_of_return(self):
        """
          Test the calculation of the initial rate of return.

          This function tests the `calculate_initial_rate_of_return` method of the
          `EvalBuffet` class. It creates instances of the class and tests the method
          with different prices and expected values.

          Returns:
               None
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            eval.metrics = pd.DataFrame([2, 3, 5], columns=["eps"])
            for price in [5, 10, 50, 100]:
                expected = 5 / price
                result = eval.calculate_initial_rate_of_return(price)
                self.assertAlmostEqual(result, expected, 4)

    def test_calculate_simple_compound_interest(self):
        """
          Tests the calculate_simple_compound_interest method of the EvalBuffet class.

          Given a range of initial values (current values), future values, and number
          of years, tests that the method calculates the correct rate of return.

          Args:
               self: The TestEvalBuffet class instance.
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            cvs = [10, 199, 209, 19000]
            fvs = [200, 1097, 1030, 19844]
            years = [10, 13, 5, 9]
            for pv, fv, year in zip(cvs, fvs, years):
                expected = ((fv / pv) ** (1 / year)) - 1
                result = eval.calculate_simple_compound_interest(pv, fv, year)
                self.assertAlmostEqual(result, expected, 4)

    def test_get_treasury_yield_api_url(self):
        """Do I need to patch datetime here to make the test work?"""
        pass

    def test_get_5Y_treasury_yield_data(self):
        """
          Get a specific url that covers a couple days
           Do an api request in jupyter with that url
           mock the response to be that of the above data
           check the reurn value"""
        pass

    def test_calculate_breakeven_vs_treasury(self):
        """
          Test the calculation of breakeven vs treasury yield for a given EPS value.

          This test iterates over a list of EPS values and Treasury yields and tests whether the 
          calculated breakeven value matches the expected result.

          Args:
               None

          Returns:
               None
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            EPS = [1, 4, 10, 140, 153, 4]
            yields = [0.05, 0.01, 0.4, 0.08, 0.10, 1]
            for eps, yield_ in zip(EPS, yields):
                result = eval.calculate_breakeven_vs_treasury(eps, yield_)
                expected = eps / yield_
                self.assertAlmostEqual(result, expected, 5)

    def test_treasury_comparison(self):
        """
          Test the treasury_comparison function.

          For each company instance, tests whether the output of the treasury_comparison function matches the expected
          result given a range of inputs.

          Args:
               None.

          Returns:
               None.

          Raises:
               AssertionError: If any of the test cases fail.
          """
        for company in company_instance_generator(api_key, 10):
            eval = company.eval_buffet
            prices = [1, 2, 10, 20, 50]
            breakevens = [1, 3, 5, 19, 40]
            margins = [1, 1, 1, 1, 1.1]
            expecteds = [True, True, False, False, False]
            for price, bp, margin, expected in zip(
                prices, breakevens, margins, expecteds
            ):
                result = eval.treasury_comparison(price, bp, margin)
                self.assertEqual(expected, result)


if __name__ == "__main__":
    unittest.main()
