import unittest
import sys
sys.path.append("..")
from investment_tools import Company
import numpy as np
from pathlib import Path
import itertools

key_path = Path().home()/'desktop'/'FinancialModellingPrep_API.txt'
with open(key_path) as file:
    api_key = file.read()

class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # cls.tickers = ['AAPL', 'MSFT', 'NVDA','VAC', 'WBA', 'ATVI', 'A', 'AMD']
        cls.tickers = ['AAPL']
        cls.api_key = api_key
        cls.data =    ['online', 'local']
        cls.period =  ['annual', 'quarter']
        cls.limit = 15
        cls.zipped_args_tdp = list(itertools.product(cls.tickers, cls.data, cls.period))

    def test_get_modifier(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for string in ['deb', 'ebt', 'dbt', 'det']:
                expected = 1
                result = instance.eval.get_modifier(string)
                self.assertEqual(expected, result)
            
            for string in ['debt', 'DeBt', 'DEBT', 'DEbT']:
                expected = -1
                result = instance.get_modifier(string)
                self.assertEqual(expected, result)
    


    def test_get_scoring_metrics(self):
        pass

    def test_create_scoring_metrics_results_dict(self):
        pass

    def test_score_single_metric(self):
        pass

    def test_get_modifier(self):
        pass


    def test_get_copy_of_df_column(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for string in ['eps', 'ebitdaratio', 'ROIC', 'totalDebtRatio']:
                expected = instance.eval.metrics[string].copy().dropna()
                fetched = instance.eval.get_copy_of_df_column(string)
                result = expected.equals(fetched)
                self.assertEqual(result, True)
    
    def test_get_r2_val(self):
        series_dict = {
            '1': [0,1,2,3,4,5],
            '0.9888': [2, 4, 6, 9],
            '0.9704': [1, 3, 4, 6, 9],
            '0.9730': [0, -1, -2, -3, -5],
            '0.9795': [0, -3, -4, -7, -10]
        }
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for r2, series in series_dict.items():
                r2_ = float(r2)
                expected = instance.eval.get_r2_val(series)
                self.assertAlmostEqual(expected, r2_, places=3)

    def test_score_mean_growth(self):
        mean_growth_score_tuple = [(0.05, 0), (0.051, 1), (0.099, 1), 
                                    (0.1001, 2), (0.1499, 2), (0.15001, 3),
                                    (0.2, 3), (0.2001, 4)]
        
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)

            for mean, score in mean_growth_score_tuple:
                self.assertEqual(instance.eval.score_mean_growth(mean), score)

    def test_score_trend_strength(self):
        r2_to_score_tuples = [(0.01, 0), (0.2, 0), (0.2001, 1), (0.3, 1), (0.3001, 2),
                              (0.5, 2), (0.5001, 3), (0.75, 3), (0.751, 4)]
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            for r2, score in r2_to_score_tuples:
                self.assertEqual(instance.eval.score_trend_strength(r2), score)

    def test_get_slope_and_intercept(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            arrays = [np.array([1, 2, 3, 5, 7, 9]),
                      np.array([1, 2, 3, 5, 8, 9]),
                      np.array([1, 2, 3, 5, 8, 13])
                     ]
            expected = [(1.6285714285714286, 0.4285714285714288),
                        (1.7142857142857144, 0.3809523809523805),
                        (2.2857142857142856, -0.3809523809523805)
                        ]      
            for array, expected_ in zip(arrays, expected):
                result = instance.eval.get_slope_and_intercept(array)
                self.assertAlmostEqual(expected_[0], result[0], 4)
                self.assertAlmostEqual(expected_[1], result[1], 4)

    def test_calculate_mean_growth_rate(self):
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            arrays = [np.array([1, 2, 3, 5, 7, 9]),
                      np.array([1, 2, 3, 5, 8, 9]),
                      np.array([2, 2, 3, 5, 8, 13])
                     ]
            expected = [0.8205642030260802,
                        0.880241233365431,
                        1.3777309915742477
                        ]      
            for array, expected_ in zip(arrays, expected):
                result = instance.eval.calculate_mean_growth_rate(array)
                self.assertAlmostEqual(expected_, result, 2)
                self.assertAlmostEqual(expected_, result, 2)

    def test_sum_of_scoring_metric_dict_scores(self):
        pass

    def test_total_score_to_bool(self):
        pass

    def test_standard_eval(self):
        '''Needs to be changed as the fucntion has also changed'''
        for ticker, data, period in self.zipped_args_tdp:
            instance = Company(ticker, self.api_key, data, period, self.limit)
            scores = [
                {'eps': {'score': 0, 'strength': 0},
                 'returnOnEquity': {'score': 0, 'strength': 0},
                 'ROIC': {'score': 0, 'strength': 0},
                 'returnOnAssets': {'score': 0, 'strength': 0},
                 'debtToTotalCap': {'score': 0, 'strength': 4},
                 'totalDebtRatio': {'score': 0, 'strength': 4}},
                 {'eps': {'score': 4, 'strength': 0},
                 'returnOnEquity': {'score': np.nan, 'strength': 0},
                 'ROIC': {'score': np.nan, 'strength': 0},
                 'returnOnAssets': {'score': 4, 'strength': 0},
                 'debtToTotalCap': {'score': 4, 'strength': 4},
                 'totalDebtRatio': {'score': 4, 'strength': 4}},
                 {'eps': {'score': np.nan, 'strength': 0},
                 'returnOnEquity': {'score': np.nan, 'strength': 0},
                 'ROIC': {'score': np.nan, 'strength': 0},
                 'returnOnAssets': {'score': 0, 'strength': 0},
                 'debtToTotalCap': {'score': 0, 'strength': 4},
                 'totalDebtRatio': {'score': 0, 'strength': 4}}
            ]

            evals = [False, True, False]
            for score_dict, expected in zip(scores, evals):
                instance.eval.standard_scores_dict = score_dict
                result = instance.eval.standard_eval()
                self.assertEqual(result, expected)

    def test_buffet_eval(self):
        pass

if __name__ == '__main__':
    unittest.main()