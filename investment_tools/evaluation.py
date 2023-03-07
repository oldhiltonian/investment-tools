import datetime as dt
import yfinance as yf
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter
from scipy.stats import linregress
import requests
import pandas as pd
from pathlib import Path
import os
import time
from typing import Dict, Tuple, List
import pyarrow as pa
import math
from .financial_data import FinancialData
from .plots import Plots
from .manual_analysis import ManualAnalysis
from urllib.request import urlopen
import json

yf.pdr_override()

class StandardEvaluation:
    """
    A class to standardize and evaluate the financial data for a given company.

    Attributes:
        ticker (str): The stock symbol for the company.
        api_key (str): The API key for the financial data provider.
        metrics (pd.Series): A Pandas Series of financial data for the company.
        _financial_data (pd.DataFrame): A Pandas DataFrame of the financial data for the company.
        _scoring_metrics (List[str]): A list of financial metrics used to calculate scores.
        standard_scores_dict (Dict[str, dict]): A dictionary containing scores and strengths for each financial metric.
        standard_outcome (bool): A boolean indicating whether the company has favorable financial metrics based on the provided scores.

    Methods:
        get_scoring_metrics(self) -> List[str]: 
            Returns a list of financial metrics used to calculate scores.
            
        create_scoring_metrics_results_dict(self, scoring_metrics: List[str]) -> Dict[str, dict]: 
            Creates a dictionary containing scores and strengths for each financial metric.
            
        score_single_metric(self, metric: str) -> Tuple[int, int]: 
            Calculates the growth score and trend stability score of the given financial metric.
            
        get_r2_val(self, metrics: pd.Series) -> float: 
            Calculates and returns the R-squared value of a linear regression model for the given Pandas series of metrics.
            
        get_modifier(self, metric: str) -> int: 
            Returns a modifier based on the provided metric.
            
        get_copy_of_df_column(self, header: str) -> pd.Series: 
            Returns a copy of a pandas Series for a given column header in the calculated metrics DataFrame.
            
        calculate_mean_growth_rate(self, df: pd.DataFrame, span: int=None) -> float: 
            Calculates the mean growth rate of a metric over a specified time span.
            
        get_slope_and_intercept(self, df: pd.DataFrame) -> Tuple[float, float]: 
            Calculates and returns the slope and intercept of a linear regression model for the given Pandas DataFrame.
            
        score_mean_growth(self, mean_growth: float) -> int: 
            Scores the mean growth of a metric based on a set of ranges.
            
        score_trend_strength(self, r2: float) -> int: 
            Calculate the trend strength score for a financial metric.
            
        sum_of_scoring_metric_dict_scores(self, scores_dict: Dict[str, dict]) -> int: 
            Calculates the total score for a dictionary of scores and strengths.
            
        total_score_to_bool(self, total_score: int, threshold: int=None) -> bool: 
            Determines whether the total score meets a threshold and returns a boolean.
            
        standard_eval(self) -> bool: 
            Determines if the company has favorable financial metrics based on the provided scores.
    """
    def __init__(self, ticker: str, api_key: str, metrics: pd.Series,
                 financial_data: pd.DataFrame) -> None:
        """
        Initializes a StandardEvaluation object.

        Args:
            ticker (str): A string representing the stock ticker of the company.
            api_key (str): A string representing the API key to use for financial data retrieval.
            metrics (pd.Series): A Pandas series of financial metrics for the company.
            financial_data (pd.DataFrame): A Pandas dataframe of the financial data for the company.

        Returns:
            None
        """
        self.ticker = ticker
        self.api_key = api_key
        self.metrics = metrics
        self._financial_data = financial_data
        self._scoring_metrics = self.get_scoring_metrics()
        self.standard_scores_dict = \
            self.create_scoring_metrics_results_dict(self._scoring_metrics)
        self.standard_outcome = self.standard_eval()
        
    
    def get_scoring_metrics(self) -> List[str]:
        """
        Returns a list of financial metrics to score.

        Returns:
            List[str]: A list of strings representing financial metrics.
        """
        scoring_metrics = [
            "eps", "returnOnEquity", "ROIC", "returnOnAssets",
            "debtToTotalCap","totalDebtRatio"
        ]
        return scoring_metrics
    
    def create_scoring_metrics_results_dict(self, scoring_metrics: List[str]) -> Dict[str, dict]:
        """
        Creates and returns a dictionary of scores and strengths for the provided list of scoring metrics.

        Args:
        scoring_metrics (List[str]): A list of the scoring metrics to create the dictionary for.

        Returns:
        Dict[str, dict]: A dictionary of scores and strengths for the provided scoring metrics.
        """
        scores = dict()
        for metric in scoring_metrics:
            score, strength = self.score_single_metric(metric)
            scores[metric] = {"score": score, "strength": strength}
        return scores
        
    def score_single_metric(self, metric: str) -> Tuple[int, int]:
            """
            Calculates the growth score and the trend stability score of the given financial metric.
            
            Args:
            metric (str): The name of the financial metric to score.
            
            Returns:
            A tuple containing the growth score and the trend stability score.
            The growth score is an integer value between 0 and 4, indicating the mean growth rate of the metric.
            The trend stability score is an integer value between 0 and 4, indicating the strength of the trend of the metric.
            """
            modifier = self.get_modifier(metric)
            metric_ = self.get_copy_of_df_column(metric)
            mean_growth = self.calculate_mean_growth_rate(metric_)
            print(mean_growth)
            r2 = self.get_r2_val(metric_)
            growth_score = self.score_mean_growth(modifier * mean_growth)
            stability_score = self.score_trend_strength(r2)
            return (growth_score, stability_score)
    

    def get_r2_val(self, metrics: pd.Series) -> float:
        """
        Calculates and returns the R-squared value of a linear regression model for the given Pandas series of metrics.

        Args:
        metrics (pd.Series): A Pandas series containing the metric values.

        Returns:
        float: The R-squared value of the linear regression model.

        """
        try:
            slope, intercept, r_val, _, _ = linregress(range(len(metrics)), metrics)
            return r_val ** 2
        except ValueError:
            return 0.0

    def get_modifier(self, metric: str) -> int:
        """
        Returns a modifier based on the provided metric. If the metric name contains
        'debt', the modifier is -1. Otherwise, it is 1.

        Args:
        metric (str): The name of the metric.

        Returns:
        int: The modifier to be used in calculations.
        """
        if "debt" in metric.lower():
            return -1
        else:
            return 1

    def get_copy_of_df_column(self, header: str) -> pd.Series:
        """
        Returns a copy of a pandas Series for a given column header in the calculated metrics DataFrame.

        Args:
        header (str): The header of the column to return.

        Returns:
        pd.Series: A copy of the column as a pandas Series with any missing values removed.
        """
        return self.metrics[header].copy().dropna()
    
    def calculate_mean_growth_rate(self, df: pd.DataFrame, span: int=None) -> float:
        """
        Calculates the mean growth rate of a financial metric over a specified span of time.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the financial metric data.
            span (int, optional): The number of periods over which to calculate the growth rate. 
                If not provided, calculates the growth rate over the full length of the DataFrame.

        Returns:
            float: The mean growth rate of the financial metric over the specified span of time.

        """
        df_ = df.iloc[-int(span)-1:] if span else df
        slope, intercept = self.get_slope_and_intercept(df_)
        x = range(len(df_))
        y = slope*x + intercept
        start, end = y[0], y[-1]
        if end <= start:
            return 0
        mean_growth = ((end/start)**(1/(len(x)-1)) - 1)
        return(mean_growth)
    
    def get_slope_and_intercept(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculates the slope and intercept of a linear regression line for the given pandas DataFrame.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the data to calculate the slope and intercept from.

        Returns:
            Tuple[float, float]: A tuple containing the calculated slope and intercept as floats with rounding to 4 decimal places.
        """
        slope, intercept, _, _, _ = linregress(range(len(df)), df)
        return round(slope,4), round(intercept,4)

    def score_mean_growth(self, mean_growth: float) -> int:
        """
        Scores the mean growth of a metric based on the following ranges:

        - 0: <= 0.05
        - 1: <= 0.10
        - 2: <= 0.15
        - 3: <= 0.20
        - 4: > 0.20

        Args:
            mean_growth (float): The mean growth of a financial metric

        Returns:
            int: The score for the mean growth of the metric based on the above ranges.
        """
        growth = float(mean_growth)
        if math.isnan(growth):
            return np.nan
        elif growth <= 0.05:
            return 0
        elif growth <= 0.10:
            return 1
        elif growth <= 0.15:
            return 2
        elif growth <= 0.2:
            return 3
        else:
            return 4
        
    def score_trend_strength(self, r2: float) -> int:
        """
        Calculate the trend strength score for a financial metric.

        Args:
            r2 (float): The coefficient of determination (R^2) of the trend line.

        Returns:
            int: The trend strength score for the metric, from 0 to 4.

        """
        r2_ = float(r2)
        if math.isnan(r2):
            return np.nan
        elif r2_ <= 0.2:
            return 0
        elif r2_ <= 0.3:
            return 1
        elif r2_ <= 0.5:
            return 2
        elif r2_ <= 0.75:
            return 3
        else:
            return 4
        
    def sum_of_scoring_metric_dict_scores(self, scores_dict: Dict[str, dict]) -> int:
        """
        Sums the scores for each metric in a given dictionary of scores and strengths.

        Args:
            scores_dict (Dict[str, dict]): A dictionary of scores and strengths for various financial metrics.

        Returns:
            int: The sum of the scores for each metric in the given dictionary. If a score is NaN, it is treated as 0.

        """
        total_score = 0
        for key in scores_dict.keys():
            score = scores_dict[key]['score']
            if math.isnan(score):
                total_score += 0
            else:
                total_score += scores_dict[key]["score"]
        return total_score

    def total_score_to_bool(self, total_score: int, threshold: int=None) -> bool:
        """
        Determines if the total score of the scoring metrics meets the threshold.

        Args:
            total_score (int): The total score of the scoring metrics.
            threshold (int, optional): The threshold to meet or exceed. Defaults to 2*len(self._scoring_metrics).

        Returns:
            bool: True if the total score meets or exceeds the threshold, False otherwise.
        """
        threshold = threshold if threshold else 2*len(self._scoring_metrics)
        return True if total_score >= threshold else False

    def standard_eval(self) -> bool:
        """
        Determines if the company has favorable financial metrics based on the provided scores.

        Args:
            None

        Returns:
            bool: True if the company has favorable financial metrics based on the provided scores, False otherwise.
        """
        total_score = self.sum_of_scoring_metric_dict_scores(self.standard_scores_dict)
        bool_result = self.total_score_to_bool(total_score)
        return bool_result
    



class BuffetEvaluation(StandardEvaluation):
    '''
    This class is under active development
    '''
    def __init__(self, ticker: str, api_key: str, metrics: pd.Series,
                 financial_data: pd.DataFrame) -> None:
        super().__init__(ticker, api_key, metrics, financial_data)
        self.buffet_outcome = self.buffet_eval()

    def buffet_eval(self) -> bool:
        print('Entering Evaluation.buffet_eval()')
        # 1. is eps increasing reliably?
        eps_growth_score = self.standard_scores_dict['eps']['score']

        # 2. determine the initial rate of return as the current price over known eps
                # higher price == lower initial rate of return
        current_stock_price = self.get_x_day_mean_stock_price(10)
        initial_fractional_return = self.calculate_initial_rate_of_return(
                                        current_stock_price)
        print('initial rate of return', initial_fractional_return)

        # 3. determine the per share growth rate over 5, 7 and 10 years
                # compounding from present - probably use the linregress instead
                #   of the actual values. 
        eps = self.metrics['eps']
        growth_rate_3_years = self.calculate_mean_growth_rate(eps, 3)
        growth_rate_5_years = self.calculate_mean_growth_rate(eps, 5)
        growth_rate_7_years = self.calculate_mean_growth_rate(eps, 7)
        growth_rate_10_years = self.calculate_mean_growth_rate(eps, 10)
        print('growth_rates for 3, 5, 7, 10 years')
        print(growth_rate_3_years)
        print(growth_rate_5_years)
        print(growth_rate_7_years)
        print(growth_rate_10_years)
        

        # 3. determine the value of the compnay relative to government bonds
            # maybe get from fmp-economics-treasury rates

        treasury_yield_5Y = self.get_5Y_treasury_yield_data(
                                        self.get_treasury_yield_api_url())
        print(treasury_yield_5Y)

        breakeven_price_vs_5Y_treasury = self.calculate_breakeven_vs_treasury(
                                                eps.iloc[-1],
                                                treasury_yield_5Y
                                                )

        bool_better_than_treasury = self.treasury_comparison(current_stock_price,
                                                             breakeven_price_vs_5Y_treasury,
                                                             1.1)
        
        print(bool_better_than_treasury)

        # 4. Determine projected annual compounding rate of return Part 1
            # project the per-share equity value for 5-10 years
            # multiply that by the projected per-share future RoE
            # this gives future predicted eps
            # then calculate the compounding rate of return
        
        equity_per_share = self.metrics['shareholderEquityPerShare']
        current_equity_per_share = equity_per_share.iloc[-1]
        average_roe_7y = self.calculate_mean_growth_rate(equity_per_share, 7)
        future_equity_per_share = self.project_future_value(current_equity_per_share,
                                                            average_roe_7y,
                                                            7)
        discounted_15pct_present_value = self.simple_discount_to_present(future_equity_per_share, 7)
        print('current equity/share', current_equity_per_share)
        print('average roe 7Y', average_roe_7y)
        print('future equity per share', future_equity_per_share)
        print('future equity discounted at 15%', discounted_15pct_present_value)
        
        # 5. Determine future per share earnings and per share stock price
        average_payout_ratio_7Y = self.metrics['dividendPayoutRatio'].iloc[-7:].mean()
        average_roe_7Y = self.metrics['returnOnEquity'].iloc[-7:].mean()
        retained_equity_pct = average_roe_7Y*(1-average_payout_ratio_7Y)
        future_equity_per_share2 = self.project_future_value(current_equity_per_share,
                                                            retained_equity_pct,
                                                            7)
        print('average payout ratio 7Y', average_payout_ratio_7Y)
        print('average roe 7y', average_roe_7Y)
        print('retained equity pct', retained_equity_pct)
        print('future equity per share incl payouts', future_equity_per_share2)
        future_eps = average_roe_7Y*future_equity_per_share2
        average_PE_low = self.metrics['PE_low'].iloc[-7:].mean()
        average_PE_high = self.metrics['PE_high'].iloc[-7:].mean()

        print('future eps', future_eps)
        print('average PE low', average_PE_low)
        print('average PE high', average_PE_high)                    
        




        pass


    def project_future_value(self, current_value: float, rate: float, years: int) -> float:
        return current_value * (1+rate)**years
    
    def simple_discount_to_present(self, future, years, rate=0.15):
        return future/((1+rate)**years)
        


    def get_x_day_mean_stock_price(self, days: int=30) -> float:
        start_date = dt.datetime.now() - dt.timedelta(int(days))
        price = pdr.get_data_yahoo(self.ticker, start=start_date)['Close'].mean()
        return price
    
    def calculate_initial_rate_of_return(self, price: float) -> float:
        latest_eps = self.metrics['eps'][-1]
        return latest_eps/price
    
    def get_treasury_yield_api_url(self):
        fmp_template = "https://financialmodelingprep.com/api/v4/treasury?from={}&to={}&apikey={}"
        from_ = str(dt.date.today() - dt.timedelta(180))
        to = str(dt.date.today())
        return fmp_template.format(from_, to, self.api_key)
    
    def get_5Y_treasury_yield_data(self, url):
        """
        Receive the content of ``url``, parse it as JSON and return the object.

        Parameters
        ----------
        url : str

        Returns
        -------
        dict
        """
        response = urlopen(url)
        data = response.read().decode("utf-8")
        return (json.loads(data)[0]['year5'])/100
    
    def calculate_breakeven_vs_treasury(self, eps, treasury_tield):
        print('breakeven price', eps/treasury_tield)
        return eps/treasury_tield

    def treasury_comparison(self, stock_price, breakeven_price, margin):
        # breakeven price neglects the fact that bonds are pre-tax and eps is post-tax
        # and it also excludes the growth rate of the stock
        return True if stock_price <= margin*breakeven_price else False #1.1 for close calls
