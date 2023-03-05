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

yf.pdr_override()

class ManualAnalysis:
    """
    Class for performing manual financial analysis.

    Args:
        financial_data (FinancialData): Object containing all financial data.
        verbose (bool, optional): Whether to print verbose messages during analysis. Defaults to False.

    Attributes:
        data (FinancialData): Object containing all financial data.
        verbose (bool): Whether to print verbose messages during analysis.
        calculated_metrics (pd.DataFrame): Contains all calculated financial metrics and ratios.
        calculation_error_dict (Dict): Contains the error count and message for each financial metric.
        fractional_metric_errors (pd.DataFrame): Contains fractional errors for each financial metric.

    Methods:
        print_metric_errors(metric_errors, tolerance=0.05):
            Prints the number of values for each metric that exceed the given tolerance level.
        assert_non_null_frame(df):
            Asserts that a given dataframe is not null. Raises an AssertionError if the dataframe is null.
        concat_stock_eval_ratios(df):
            Concatenates and returns a dataframe containing stock evaluation ratios.
        concat_profitability_ratios(df):
            Concatenates and returns a dataframe containing profitability ratios.
        concat_debt_interest_ratios(df):
            Concatenates and returns a dataframe containing debt and interest ratios.
        concat_liquidity_ratios(df):
            Concatenates and returns a dataframe containing liquidity ratios.
        concat_efficiency_ratios(df):
            Concatenates and returns a dataframe containing efficiency ratios.
        concat_metric_growth(df):
            Concatenates and returns a dataframe containing the growth rate for each financial metric.
        analyse() -> pd.DataFrame:
            Returns a dataframe containing all calculated financial metrics and ratios.
        cross_check_metric_calculations() -> pd.DataFrame:
            Cross-checks reported financial metrics against calculated financial metrics and returns 
            the fractional errors.

    """

    def __init__(self, financial_data: FinancialData, verbose: bool = False) -> None:
        """
        Initializes the ManualAnalysis object.

        Args:
            financial_data (FinancialData): Object containing all financial data.
            verbose (bool, optional): Whether to print verbose messages during analysis. Defaults to False.

        Raises:
            AssertionError: If any of the calculated metrics are null.
        """
        self.data = financial_data
        self.verbose = verbose
        self.calculated_metrics = self.analyse()
        # print(self.calculated_metrics)
        # self.assert_non_null_frame(self.calculated_metrics)
        self.calculation_error_dict = {}
        if self.verbose:
            self.fractional_metric_errors = self.cross_check_metric_calculations()
            # self.assert_non_null_frame(self.fractional_metric_errors)
            self.print_metric_errors(self.fractional_metric_errors, 0.05)

    def print_metric_errors(self, metric_errors: pd.DataFrame, tolerance: float = 0.05):
        """
        Prints the number of values for each metric that exceed the given tolerance level.

        Args:
            metric_errors (pd.DataFrame): Contains the fractional errors for each financial metric.
            tolerance (float, optional): The tolerance level for the error fraction. Defaults to 0.05.
        """
        if not self.verbose:
            return
        line_count = len(metric_errors)
        for header in metric_errors:
            count = sum(metric_errors[header] >= tolerance)
            message = f"There were {count}/{line_count} values in {header} that exceed the {tolerance} error tolerance."
            self.calculation_error_dict[header] = (count, message)
            print(message)

    def assert_non_null_frame(self, df: pd.DataFrame):
        """
        Raises an AssertionError if any column in the input DataFrame is completely null.

        Args:
            df (pandas.DataFrame): A pandas DataFrame.

        Raises:
            AssertionError: If any column in `df` is completely null.

        """
        err_msg = "Metric error calculations returned a null dataframe. Unreliable calculations."
        for header in df.columns:
            assert not df[header].isnull().all(), f"<{header}> -- {err_msg}"

    def concat_stock_eval_ratios(self, df: pd.DataFrame):
        """
        Calculate and concatenate stock evaluation ratios to the given DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to concatenate the calculated metrics to.

        Returns:
            pandas.DataFrame: The resulting DataFrame containing the calculated metrics.

        Calculates the following stock evaluation ratios:
            - Earnings per share (EPS)
            - Diluted earnings per share (EPSDiluted)
            - Price to earnings ratio (PE) for high, low, and average closing stock prices
            - Book value per share
            - Dividend payout ratio
            - Dividend yield for low, high, and average closing stock prices
            - EBITDA ratio
            - Cash per share

        All ratios are calculated using financial data from the associated FinancialData object.
        """
        total_assets = self.data.balance_sheets["totalAssets"].copy()
        total_liabilities = self.data.balance_sheets["totalLiabilities"].copy()
        long_term_debt = self.data.balance_sheets["longTermDebt"].copy()
        dividends_paid = self.data.cash_flow_statements["dividendsPaid"].copy()
        outstanding_shares = self.data.income_statements[
            "outstandingShares_calc"
        ].copy()
        cash_and_equivalents = self.data.balance_sheets["cashAndCashEquivalents"].copy()
        eps = self.data.income_statements["eps"].copy()
        stock_price_high = self.data.stock_price_data["high"].copy()
        stock_price_avg = self.data.stock_price_data["avg_close"].copy()
        stock_price_low = self.data.stock_price_data["low"].copy()
        df["eps"] = eps  # authorized stock!!!
        df["eps_diluted"] = self.data.income_statements["epsdiluted"]
        df["PE_high"] = stock_price_high / eps
        df["PE_low"] = stock_price_low / eps
        df["PE_avg_close"] = stock_price_avg / eps
        df["bookValuePerShare"] = (
            total_assets - total_liabilities
        ) / outstanding_shares
        df["dividendPayoutRatio"] = (-dividends_paid / outstanding_shares) / eps
        df["dividendYield_low"] = (
            -dividends_paid / outstanding_shares
        ) / stock_price_high
        df["dividendYield_high"] = (
            -dividends_paid / outstanding_shares
        ) / stock_price_low
        df["dividendYield_avg_close"] = (
            -dividends_paid / outstanding_shares
        ) / stock_price_avg
        df["ebitdaratio"] = self.data.income_statements["ebitdaratio"]
        df["cashPerShare"] = (
            1 * (cash_and_equivalents - long_term_debt)
        ) / outstanding_shares
        return df

    def concat_profitability_ratios(self, df: pd.DataFrame):
        """
        Concatenates profitability ratios to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to which profitability ratios will be added.

        Returns:
            pd.DataFrame: DataFrame with added profitability ratios.

        Calculates the following profitability ratios:
            - Gross profit margin
            - Operating profit margin
            - Pretax profit margin
            - Net profit margin
            - Return on invested capital (ROIC)
            - Return on equity (ROE)
            - Return on Assets (ROA)

        All ratios are calculated using financial data from the associated FinancialData object.
        """
        revenue = self.data.income_statements["revenue"].copy()
        total_assets = self.data.balance_sheets["totalAssets"].copy()
        gross_profit = self.data.income_statements["grossProfit"].copy()
        operating_income = self.data.income_statements["operatingIncome"].copy()
        income_before_tax = self.data.income_statements["incomeBeforeTax"].copy()
        total_capitalization = (
            self.data.balance_sheets["totalEquity"].copy()
            + self.data.balance_sheets["longTermDebt"].copy()
        )
        net_income = self.data.income_statements["netIncome"].copy()
        total_shareholder_equity = self.data.balance_sheets[
            "totalStockholdersEquity"
        ].copy()
        df["grossProfitMargin"] = gross_profit / revenue
        df["operatingProfitMargin"] = operating_income / revenue
        df["pretaxProfitMargin"] = income_before_tax / revenue
        df["netProfitMargin"] = net_income / revenue
        df["ROIC"] = net_income / total_capitalization
        df["returnOnEquity"] = net_income / total_shareholder_equity
        df["returnOnAssets"] = net_income / total_assets
        return df

    def concat_debt_interest_ratios(self, df: pd.DataFrame):
        """
        Concatenates debt and interest ratios to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to which debt and interest ratios will be added.

        Returns:
            pd.DataFrame: DataFrame with added debt and interest ratios.

        Calculates the following debt and interest ratios:
            - Interest coverage
            - Fixed charge coverage
            - Debt to total capitalization ratio
            - Total debt ratio

        All ratios are calculated using financial data from the associated FinancialData object.
        """
        operating_income = self.data.income_statements["operatingIncome"].copy()
        total_assets = self.data.balance_sheets["totalAssets"].copy()
        long_term_debt = self.data.balance_sheets["longTermDebt"].copy()
        total_capitalization = (
            self.data.balance_sheets["totalEquity"].copy()
            + self.data.balance_sheets["longTermDebt"].copy()
        )
        interest_expense = self.data.income_statements["interestExpense"].copy()
        # The fixed_charges calculation below is likely incomplete
        fixed_charges = (
            self.data.income_statements["interestExpense"].copy()
            + self.data.balance_sheets["capitalLeaseObligations"].copy()
        )
        ebitda = self.data.income_statements["ebitda"].copy()
        total_equity = self.data.balance_sheets["totalEquity"].copy()
        total_debt = total_assets - total_equity
        df["interestCoverage"] = operating_income / interest_expense
        df["fixedChargeCoverage"] = ebitda / fixed_charges
        df["debtToTotalCap"] = long_term_debt / total_capitalization
        df["totalDebtRatio"] = total_debt / total_assets
        return df

    def concat_liquidity_ratios(self, df: pd.DataFrame):
        """
        Concatenates liquidity ratios to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to which liquidity ratios will be added.

        Returns:
            pd.DataFrame: DataFrame with added liquidity ratios.

        Calculates the following liquidity ratios:
            - Current ratio
            - Quick ratio
            - Cash ratio

        All ratios are calculated using financial data from the associated FinancialData object.
        """
        current_assets = self.data.balance_sheets["totalCurrentAssets"].copy()
        current_liabilities = self.data.balance_sheets["totalCurrentLiabilities"].copy()
        inventory = self.data.balance_sheets["inventory"].copy()
        cash_and_equivalents = self.data.balance_sheets["cashAndCashEquivalents"].copy()
        quick_assets = current_assets - inventory
        df["currentRatio"] = current_assets / current_liabilities
        df["quickRatio"] = quick_assets / current_liabilities
        df["cashRatio"] = cash_and_equivalents / current_liabilities
        return df

    def concat_efficiency_ratios(self, df: pd.DataFrame):
        """
        Concatenates efficiency ratios to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to which efficiency ratios will be added.

        Returns:
            pd.DataFrame: DataFrame with added efficiency ratios.

        Calculates the following efficiency ratios:
            - Total asset turnover
            - Inventory to sales ratio
            - Inventory turnover ratio
            - Inventory turnover in days
            - Accounts receivable to sales ratio
            - Receivables turnover
            - Receivables turnover in days

        All ratios are calculated using financial data from the associated FinancialData object.
        """
        inventory = self.data.balance_sheets["inventory"].copy()
        revenue = self.data.income_statements["revenue"].copy()
        total_assets = self.data.balance_sheets["totalAssets"].copy()
        net_receivables = self.data.balance_sheets["netReceivables"].copy()
        df["totalAssetTurnover"] = revenue / total_assets
        df["inventoryToSalesRatio"] = inventory / revenue
        df["inventoryTurnoverRatio"] = 1 / df["inventoryToSalesRatio"]
        days = 365 if self.data.period == "annual" else 90
        df["inventoryTurnoverInDays"] = days / df["inventoryTurnoverRatio"].copy()
        accounts_receivable_to_sales_ratio = (
            self.data.balance_sheets["netReceivables"].copy() / revenue
        )
        df["accountsReceivableToSalesRatio"] = accounts_receivable_to_sales_ratio
        df["receivablesTurnover"] = revenue / net_receivables
        df["receivablesTurnoverInDays"] = days / df["receivablesTurnover"].copy()
        return df

    def concat_metric_growth(self, df: pd.DataFrame):
        """
        Concatenates growth rates for specified metrics to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to which growth rates will be added.

        Returns:
            pd.DataFrame: DataFrame with added growth rates.

        Calculates the percentage growth rate for the following metrics over a one year or four quarter period:
            - Earnings per share (EPS)
            - Return on equity (ROE)
            - Cash per share
            - Price-to-earnings (P/E) ratio (average close)
            - Earnings before interest, taxes, depreciation, and amortization (EBITDA) ratio
            - Return on invested capital (ROIC)
            - Net profit margin
            - Return on assets (ROA)
            - Debt to total capitalization ratio
            - Total debt ratio

        All growth rates are calculated using financial data from the associated FinancialData object.
        """
        growth_metrics = [
            "eps",
            "returnOnEquity",
            "cashPerShare",
            "PE_avg_close",
            "ebitdaratio",
            "ROIC",
            "netProfitMargin",
            "returnOnAssets",
            "debtToTotalCap",
            "totalDebtRatio",
        ]
        span, init_idx = (1, 1) if self.data.period == "annual" else (4, 4)
        for metric in growth_metrics:
            series = df[metric]
            col_header = metric + "_growth"
            col_data = []
            for idx, value in enumerate(series):
                if idx < init_idx:
                    col_data.append(np.nan)
                else:
                    col_data.append(
                        (value - series[idx - span]) / abs(series[idx - span])
                    )
            df[col_header] = pd.Series(col_data, index=self.data.frame_indecies)
        return df

    def analyse(self) -> pd.DataFrame:
        """
        Analyzes the financial data and calculates a variety of financial ratios.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated financial ratios.

        The following ratio types are calculated using financial data from the associated FinancialData object:
            - Stock evaluation ratios
            - Profitability ratios
            - Debt and interest ratios
            - Liquidity ratios
            - Efficiency ratios
            - Metric growth
        """
        df = pd.DataFrame(index=self.data.frame_indecies)
        df = self.concat_stock_eval_ratios(df)
        df = self.concat_profitability_ratios(df)
        df = self.concat_debt_interest_ratios(df)
        df = self.concat_liquidity_ratios(df)
        df = self.concat_efficiency_ratios(df)
        df = self.concat_metric_growth(df)
        return df

    def cross_check_metric_calculations(self) -> pd.DataFrame:
        """
        Calculates the fractional error between reported metrics and those calculated by the program 
        for a selection of key metrics. If verbose is set to False, returns None.

        Returns:
            pd.DataFrame: DataFrame containing the fractional errors for each metric.

        Metrics that are cross-checked:
            - Gross profit margin
            - Operating profit margin
            - Net profit margin
            - Current ratio
            - Return on equity (ROE)
            - Return on assets (ROA)
            - Cash per share
            - Interest coverage
            - Dividend payout ratio

        Fractional error is calculated as the absolute value of the difference between the calculated 
        metric and the reported metric, divided by the calculated metric.
        """
        ## add ROE, ROIC, ebitda ratio
        if not self.verbose:
            return
        fractional_errors = pd.DataFrame(index=self.data.frame_indecies)
        metrics_to_check = [
            "grossProfitMargin",
            "operatingProfitMargin",
            "netProfitMargin",
            "currentRatio",
            "returnOnEquity",
            "returnOnAssets",
            "cashPerShare",
            "interestCoverage",
            "dividendPayoutRatio",
        ]
        for metric in metrics_to_check:
            reported = self.data.reported_key_metrics[metric]
            calculated = self.calculated_metrics[metric]
            try:
                if sum(reported) == 0 and sum(calculated) == 0:
                    fractional_errors[metric] = calculated
                else:
                    fractional_errors[metric] = ((calculated - reported) / calculated).abs()
            except TypeError:
                raise TypeError(f"{metric} series has sunsupported type for sum()")
        return fractional_errors