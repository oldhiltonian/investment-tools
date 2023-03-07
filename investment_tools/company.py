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
from .evaluation import Evaluation, BuffetEvaluation

yf.pdr_override()

class Company:
    """
    Class representing a company and its financial analysis

    Args:
    ticker (str): The stock symbol representing the company
    api_key (str): API key for accessing financial data from a data source
    data (str, optional): Source of data either from 'online' (default) or 'offline'
    period (str, optional): Financial period to use for the analysis either 'annual' (default) or 'quarterly'
    limit (int, optional): The number of financial periods to include in the analysis (default is 20)
    verbose (bool, optional): If True, verbose output will be printed during analysis (default is False)

    Attributes:
    ticker (str): The stock symbol representing the company
    period (str): Financial period used for the analysis
    metrics (dict): Dictionary of financial metrics for the company
    trends (list of plot objects): List of plots showing the trend of the financial metrics over time

    Methods:
    __init__(self, ticker, api_key, data='online', period='annual', limit=20, verbose=False):
        Initializes a new instance of the `Company` class.

        Parameters:
        ticker (str): The stock symbol representing the company
        api_key (str): API key for accessing financial data from a data source
        data (str, optional): Source of data either from 'online' (default) or 'offline'
        period (str, optional): Financial period to use for the analysis either 'annual' (default) or 'quarterly'
        limit (int, optional): The number of financial periods to include in the analysis (default is 20)
        verbose (bool, optional): If True, verbose output will be printed during analysis (default is False)

    recommendation(self) -> Dict[str, dict]:
        Builds a recommendation for the company based on its financial metrics.

        Returns:
        A dictionary with the keys being the metrics and the values being another dictionary with 'score'
             and 'strength' keys.

    get_modifier(self, metric: str) -> int:
        Returns a modifier based on whether the metric is related to debt.

        Parameters:
        metric (str): The name of the financial metric.

        Returns:
        1 if the metric is not related to debt, -1 otherwise.

    get_copy_of_df_column(self, header: str) -> pd.Series:
        Returns a copy of a column of data for a given metric.

        Parameters:
        header (str): The name of the financial metric.

        Returns:
        A copy of the column of data for the given metric.

    get_r2_val(metrics: pd.Series) -> float:
        Returns the R-squared value for a given series of metrics.

        Parameters:
        metrics (pd.Series): A series of metrics.

        Returns:
        The R-squared value for the given series of metrics.

    score(self, metric: str) -> Tuple[int, int]:
        Computes the score for a given metric.

        Parameters:
        metric (str): The name of the financial metric.

        Returns:
        A tuple containing the growth score and the stability score for the given metric.

    score_mean_growth(self, mean_growth: float) -> int:
        Computes the growth score for a given mean growth rate.

        Parameters:
        mean_growth (float): The mean growth rate.

        Returns:
        The growth score for the given mean growth rate.

    score_trend_strength(r2: float) -> int:
        Calculates the trend strength score based on the R^2 value of a metric.
    
        Args:
        r2 (float): The R^2 value of a metric.
        
        Returns:
        int: The trend strength score.

    eval_(scores: Dict[str, dict]) -> bool:
        Evaluates the overall recommendation for the company based on the scores for each metric.
        
        Args:
        scores (Dict[str, dict]): A dictionary of scores for each metric.
        
        Returns:
        bool: True if the recommendation is positive, False otherwise.

    print_charts() -> None:
        Prints the trend charts for each metric.
        
        Returns:
        None

    export() -> None:
        Exports the financial trend charts to disk as a pdf file. Also exports key findings based on the analysis.
    """

    def __init__(
        self, ticker: str, api_key: str, data: str="online", period: str="annual", 
        limit: int=20, verbose: bool=False
    ):
        """
        Initialize a new instance of the Company class.

        Args:
            ticker (str): The stock symbol representing the company.
            api_key (str): API key for accessing financial data from a data source.
            data (str, optional): Source of data either from 'online' (default) or 'offline'.
            period (str, optional): Financial period to use for the analysis either 'annual' (default) or 'quarterly'.
            limit (int, optional): The number of financial periods to include in the analysis (default is 20).
            verbose (bool, optional): Set to True to enable additional console output (default is False).

        Attributes:
            ticker (str): The stock symbol representing the company.
            period (str): Financial period used for the analysis.
            metrics (dict): Dictionary of financial metrics for the company.
            trends (list of plot objects): List of plots showing the trend of the financial metrics over time.
        """
        self.ticker = ticker
        self.period = period
        self.limit = limit
        self.verbose = verbose
        self._financial_data = FinancialData(ticker, api_key, data, period, limit)
        self.filing_dates = self._financial_data.filing_date_objects
        self._analysis = ManualAnalysis(self._financial_data, self.verbose)
        self.metrics = self._analysis.calculated_metrics
        self._charts_printed = False
        if self.verbose:
            self.print_charts()
        self.eval = Evaluation(ticker, api_key, self.metrics, self._financial_data)
        self.eval_buffet = BuffetEvaluation(ticker, api_key, self.metrics, self._financial_data)
        self.standard_outcome = self.eval.standard_outcome
        if self.standard_outcome:
            self.print_charts()
            self.export()

   

    def print_charts(self) -> None:
        """
        Prints the trend charts for the company's financial metrics.
        
        If the charts have already been printed, this method does nothing.
        
        Returns:
        None
        """
        if self._charts_printed:
            return
        self._plots = Plots(
            self.ticker, self.period, self.metrics, self.limit, self.filing_dates
        )
        self.trends = self._plots.plots
        self._charts_printed = True

    def export(self) -> None:
        """
        The export() method exports the financial trend charts to a PDF file.

        Returns:
        None.
        """
        self._plots._export_charts_pdf()
        # also export key findings based on the analyis