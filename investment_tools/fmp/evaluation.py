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

yf.pdr_override()

class Evaluation:
    def __init__(self, metrics):
        self.metrics = metrics
        self._scoring_metrics = self.get_scoring_metrics()
        self.scores_dict = self.create_scoring_metrics_results_dict(self._scoring_metrics)
        self.outcome = self.eval_()
    
    def get_scoring_metrics(self):
        scoring_metrics = [
            "eps", "returnOnEquity", "ROIC", "returnOnAssets",
            "debtToTotalCap","totalDebtRatio"
        ]
        return scoring_metrics
    
    def create_scoring_metrics_results_dict(self, scoring_metrics: List) -> Dict[str, dict]:
        scores = dict()
        for metric in scoring_metrics:
            score, strength = self.score_single_metric(metric)
            scores[metric] = {"score": score, "strength": strength}
        return scores

    def create_scoring_metrics_results_dict(self, scoring_metrics: List) -> Dict[str, dict]:
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
    
    def calculate_mean_growth_rate(self, df: pd.DataFrame) -> float:
        slope, intercept = self.get_slope_and_intercept(df)
        x = range(len(df))
        y = slope*x + intercept
        start, end = y[0], y[-1]
        if end <= start:
            return 0
        mean_growth = ((end/start)**(1/len(x)) - 1)
        return(mean_growth)
    
    def get_slope_and_intercept(self, df: pd.DataFrame) -> Tuple[float, float]:
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
        
    def sum_of_scoring_metric_dict_scores(self, scores_dict: Dict[str, dict]):
        total_score = 0
        for key in scores_dict.keys():
            score = scores_dict[key]['score']
            if math.isnan(score):
                total_score += 0
            else:
                total_score += scores_dict[key]["score"]
        return total_score

    def total_score_to_bool(self, total_score: int, threshold: int=None):
        threshold = threshold if threshold else 2*len(self._scoring_metrics)
        return True if total_score >= threshold else False

    def eval_(self) -> bool:
        """
        Determines if the company has favorable financial metrics based on the provided scores.

        Args:
        scores (Dict[str, dict]): A dictionary of scores and strengths for various financial metrics

        Returns:
        bool: True if the company has favorable financial metrics based on the provided scores, False otherwise.
        
        """
        total_score = self.sum_of_scoring_metric_dict_scores(self.scores_dict)
        bool_result = self.total_score_to_bool(total_score)
        return bool_result