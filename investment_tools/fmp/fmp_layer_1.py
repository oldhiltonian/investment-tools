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
from typing import Dict, Tuple

yf.pdr_override()


class FinancialData:
    """
    A class for fetching and processing financial data for a given stock.

    Attributes:
        ticker (str): The ticker symbol for the stock.
        api_key (str): The user's API key for accessing financial data (optional).
        data (str): Whether to fetch data from online or local source. Either 'online'
            or 'local'.
        period (str): The period of the data, either 'annual' or 'quarter'.
        limit (int): The maximum number of data points to fetch from the API.
            Default = 120

    Raises:
        AssertionError: If invalid input is provided for data or period attributes.

    Methods:
        __init__(self, ticker: str, api_key: str='', data: str='local',
                 period: str='annual', limit: int=120)
            Initializes a FinancialData object.

        assert_valid_user_inputs(self)
            Asserts that the user input for the data attribute is valid.

        fetch_raw_data(self, data_type: str) -> pd.DataFrame
            Fetches raw financial data from an online source or a local cache.

        get_load_path(self, data_type: str, ticker: str, period: str) -> Path
            Generates a file path for the specified data type.

        get_frame_indecies(self) -> pd.Index
            Retrieves a list of financial data statement indecies.

        build_dataframe(self, data: dict) -> pd.DataFrame
            Builds a pandas dataframe from financial statement data.

        generate_index(self, date: str) -> str
            Generates an index for the dataframe from the filing date.

        generate_date(self, date_str: str) -> dt.date
            Converts a date string to a datetime.date object.

        check_for_matching_indecies(self) -> bool
            Asserts that financial statements have matching indecies.

        get_common_df_indicies(self) -> pd.Index
            Retrieves the common financial statement dataframe indecies.

        filter_for_common_indecies(self, common_elements: pd.Index)
            Filters financial statement dataframes for common indecies.

        assert_identical_indecies(self)
            Asserts that financial statement dataframe indecies are identical.

        assert_required_length(self, item: list)
            Asserts that a given item has the required length based on the period.

        assert_valid_server_response(Self, response: requests.Response)
            Asserts that an API call to fetch financial data was successful.

        assert_server_response_not_empty(self, response: requests.Response)
            Asserts that an API call to fetch financial data returned non-empty results.

        fetch_stock_price_data_yf(self) -> pd.DataFrame
            Fetches stock price data from Yahoo Finance.

        periodise(self, df: pd.DataFrame) -> pd.DataFrame
            Periodises the stock price data for each filing period.

        save_financial_attributes(self)
            Saves the financial data to local parquet files.
    """

    def __init__(
        self,
        ticker: str,
        api_key: str = "",
        data: str = "local",
        period: str = "annual",
        limit: int = 120,
    ):
        """
        Initializes a new instance of the FinancialData class.

        Args:
            ticker (str): The ticker symbol for the stock.
            api_key (str): The user's API key for accessing financial data (optional).
            data (str): Whether to fetch data from online or local source. Either 'online'
                or 'local'.
            period (str): The period of the data, either 'annual' or 'quarter'.
            limit (int): The maximum number of data points to fetch from the API.
                Default = 120

        """
        self.ticker = ticker.upper().strip()
        self.api_key = str(api_key)
        self.data = data.lower().strip()
        self.period = period.lower().strip()
        self.assert_valid_user_inputs()
        self.limit = int(limit)
        self.days_in_period = 365 if period == "annual" else 90

        self.balance_sheets = self.build_dataframe(self.fetch_raw_data("bs"))
        self.balance_sheets = self.replace_None(self.balance_sheets)
        self.income_statements = self.build_dataframe(self.fetch_raw_data("is"))
        self.income_statements = self.replace_None(self.income_statements)
        self.cash_flow_statements = self.build_dataframe(self.fetch_raw_data("cfs"))
        self.cash_flow_statements = self.replace_None(self.cash_flow_statements)
        self.reported_key_metrics = self.build_dataframe(self.fetch_raw_data("metrics"))
        self.reported_key_metrics = self.replace_None(self.reported_key_metrics)

        if not self.check_for_matching_indecies():
            self.filter_for_common_indecies(self.get_common_df_indicies())

        self.assert_required_length(self.balance_sheets)
        self.frame_indecies = self.get_frame_indecies()
        self.filing_date_objects = self.balance_sheets["date"]
        self.stock_price_data = self.fetch_stock_price_data_yf()
        self.assert_required_length(self.stock_price_data)
        self.save_financial_attributes()

    def replace_None(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.replace(to_replace=[None,'None'], value=0)

    def assert_valid_user_inputs(self):
        """
        Ensures that valid user inputs are provided for data and period.
        
        Raises:
            AssertionError: If invalid input is provided for data or period.
        """
        assert self.data in ["online", "local"], "data must be 'online' or 'local'"
        assert self.period in [
            "annual",
            "quarter",
        ], "period must be 'annual' or 'quarter"

    def generate_request_url(self, data_type: str) -> str:
        """
        Generates a URL to fetch data from the Financial Modeling Prep API.
        
        Args:
            data_type (str): The type of financial statement data to fetch. Valid 
                inputs are 'bs', 'is, 'cfs', and 'ratios'.
        
        Returns:
            str: The URL for the requested data.
        
        Raises:
            ValueError: If an invalid data type is provided.
        """
        fmp_template = "https://financialmodelingprep.com/api/v3/{}/{}?period={}&limit={}&apikey={}"
        ticker = self.ticker
        period = self.period
        limit = self.limit
        api_key = self.api_key

        if data_type == "bs":
            data_str = "balance-sheet-statement"
        elif data_type == "is":
            data_str = "income-statement"
        elif data_type == "cfs":
            data_str = "cash-flow-statement"
        elif data_type == "metrics":
            data_str = "ratios"
        else:
            err_msg = f"{data_type} is not a valid API call"
            raise ValueError(err_msg)

        return fmp_template.format(data_str, ticker, period, limit, api_key)

    def fetch_raw_data(self, data_type: str) -> pd.DataFrame:
        """
        Fetches raw financial data from either the Financial Modeling Prep API
          or a local file, based on the self.data attribute. 
        
        Args:
            data_type (str): The type of data to fetch.
        
        Returns:
            pandas.DataFrame if the data is fetched locally, otherwise requests.Response
        """
        if self.data == "online":
            url = self.generate_request_url(data_type)
            raw_data = requests.get(url)
            self.assert_valid_server_response(raw_data)
            self.assert_server_response_not_empty(raw_data)
        elif self.data == "local":
            path = self.get_load_path(data_type, self.ticker, self.period)
            raw_data = pd.read_parquet(path)
        return raw_data

    def get_load_path(self, data_type: str, ticker: str, period: str) -> Path:
        """
        Gets the file path to load a local financial data file.
        
        Args:
            data_type (str): The type of financial data to fetch. Acceptable 
                values are 'bs', 'is', 'cfs', and 'ratios'.
            ticker (str): The ticker symbol for the stock.
            period (str): The period of the data, either 'annual' or 'quarter'.
        
        Returns:
            pathlib.Path: The file path for the requested data.
        """
        if data_type == "bs":
            file = "balance_sheets.parquet"
        elif data_type == "is":
            file = "income_statements.parquet"
        elif data_type == "cfs":
            file = "cash_flow_statements.parquet"
        elif data_type == "metrics":
            file = "reported_key_metrics.parquet"
        elif data_type == "price":
            file = "stock_price_data.parquet"
        else:
            err_msg = f"{data_type} is not a valid API call"
            raise ValueError(err_msg)
        return (
            Path.cwd().parent
            / "data"
            / "Company_Financial_Data"
            / ticker
            / period
            / file
        )

    def get_frame_indecies(self) -> pd.Index:
        """
        Gets the index for the financial data.
        
        Returns:
            pandas.Index: The index for the financial data.
        """
        return self.balance_sheets.index

    def set_frame_index(self, other: pd.DataFrame) -> pd.DataFrame:
        """
        Sets the index for the provided DataFrame.
        
        Returns:
            pandas.DataFrame: Updated with the global frame index
        """
        other.index = self.frame_indecies
        return other

    def build_dataframe(self, data: dict) -> pd.DataFrame:
        """
        Builds a pandas.DataFrame from provided raw data. If the raw data
            is already a DataFrame then it is returned immediately. 
        
        Args:
            data (dict): The raw data to build the DataFrame from.
        
        Returns:
            pandas.DataFrame: The built DataFrame.
        """
        if self.data == "local":
            return data
        data = data.json()
        working_array = []
        for statement in reversed(data):
            working_array.append(list(statement.values()))
        df = pd.DataFrame(working_array, columns=data[0].keys())
        df["index"] = df["date"].apply(lambda x: self.generate_index(x))
        df["date"] = df["date"].apply(lambda x: self.generate_date(x))
        df = df.set_index("index")
        if "netIncome" and "eps" in df.keys():
            df["outstandingShares_calc"] = df["netIncome"] / df["eps"]
        return df

    def generate_index(self, date: str) -> str:
        """
        Generates an index for the financial data.
        
        Args:
            date (str): The date in 'YYYY-MM-DD' format.
        
        Returns:
            str: The generated index string.
        """
        year, month, _ = [int(i) for i in date.split("-")]

        if self.period == "annual":
            return f"{self.ticker}-FY-{year}"

        if month in (1, 2, 3):
            quarter = 1
        elif month in (4, 5, 6):
            quarter = 2
        elif month in (7, 8, 9):
            quarter = 3
        elif month in (10, 11, 12):
            quarter = 4

        return f"{self.ticker}-Q{quarter}-{year}"

    def generate_date(self, date_str: str) -> dt.date:
        """
        Generates a datetime.date object from a date string.
        
        Args:
            date_str (str): The date in 'YYYY-MM-DD' format.
        
        Returns:
            datetime.date: The generated date object.
        """
        year, month, day = [int(i) for i in date_str.split()[0].split("-")]
        return dt.date(year, month, day)

    def check_for_matching_indecies(self) -> bool:
        """
        Checks whether the financial data has matching indices.
        
        Returns:
            bool: True if the financial data has matching indices, False otherwise.
        """
        print("Checking matching frame indecies")
        len_bs = len(self.balance_sheets)
        len_is = len(self.income_statements)
        len_cfs = len(self.cash_flow_statements)
        len_ratios = len(self.reported_key_metrics)
        print(
            f"Financial statement lengths are BS: {len_bs}, IS:{len_is}, CFS:{len_cfs}, Ratios:{len_ratios}"
        )
        # Logical checks for matching indecies
        matching_index_1 = self.balance_sheets["date"].equals(
            self.income_statements["date"]
        )
        matching_index_2 = self.balance_sheets["date"].equals(
            self.cash_flow_statements["date"]
        )
        matching_index_3 = self.balance_sheets["date"].equals(
            self.reported_key_metrics["date"]
        )
        matching_indecies = matching_index_1 and matching_index_2 and matching_index_3
        return True if matching_indecies else False

    def get_common_df_indicies(self) -> pd.Index:
        """
        Gets the common indices for the financial data.
        
        Returns:
            pandas.Index: The common indices for the financial data.
        """
        idx1 = self.balance_sheets.index
        idx2 = self.income_statements.index
        idx3 = self.cash_flow_statements.index
        idx4 = self.reported_key_metrics.index
        common_elements = idx1.intersection(idx2).intersection(idx3).intersection(idx4)
        return common_elements

    def filter_for_common_indecies(self, common_elements: pd.Index):
        """
        Filters the financial data attributes to only include common indices.
        
        Args:
            common_elements (pandas.Index): The common indices for the financial data.
        """
        self.balance_sheets = self.balance_sheets.loc[common_elements]
        self.income_statements = self.income_statements.loc[common_elements]
        self.cash_flow_statements = self.cash_flow_statements.loc[common_elements]
        self.reported_key_metrics = self.reported_key_metrics.loc[common_elements]
        self.assert_identical_indecies()
        print(f"Financial statement lengths are now each: {len(self.balance_sheets)}")

    def assert_identical_indecies(self):
        """
        Asserts that the financial data has identical indices 
            for each of its statements.
        """
        err_msg = "Indecies could not be filtered for common elements"
        assert self.cash_flow_statements.index.equals(
            self.balance_sheets.index
        ), err_msg
        assert self.income_statements.index.equals(self.balance_sheets.index), err_msg
        assert self.reported_key_metrics.index.equals(
            self.balance_sheets.index
        ), err_msg

    def assert_required_length(self, item: list):
        """
        Asserts that a given item has the required length based on the period.
        
        Args:
            item (pandas.DataFrame): The item to check the length of.
        """
        if self.period == "annual":
            required_length = 2
        else:
            required_length = 4
        err_msg = f"Financial statements are shorter than the required length of {required_length}"
        assert len(item) >= required_length, err_msg

    def assert_valid_server_response(self, response: requests.Response):
        """
        Asserts that the server response is valid (status code 200).
        
        Args:
            response (requests.Response): The server response.
        """
        assert (
            response.status_code == 200
        ), f"API call failed. Code <{response.status_code}>"

    def assert_server_response_not_empty(self, response: requests.Response):
        """
        Asserts that the server response not empty.
        
        Args:
            response (requests.Response): The server response.
        """
        assert len(response.json()) != 0, "Server response successful but empty"

    def fetch_stock_price_data_yf(self) -> pd.DataFrame:
        """
        Fetches stock price data from Yahoo Finance.
        
        Returns:
            pandas.DataFrame: The fetched stock price data.
        """
        start_date = dt.date(
            *[
                int(i)
                for i in str(self.filing_date_objects.iloc[0]).split()[0].split("-")
            ]
        ) - dt.timedelta(days=370)
        end_date = dt.date.today()
        price_interval = "1d"
        raw_data = pdr.get_data_yahoo(
            self.ticker, start_date, end_date, interval=price_interval
        )
        raw_data["date"] = raw_data.index.date
        return self.periodise(raw_data)

    def periodise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Periodises the stock price data for each filing period.
        
        Args:
            df (pandas.DataFrame): The stock price data.
        
        Returns:
            pandas.DataFrame: The periodised stock price data.
        """
        working_array = []
        days = self.days_in_period
        for i in range(len(self.filing_date_objects)):
            if i == 0 or i == len(self.filing_date_objects):
                filter_date_end = self.filing_date_objects[i]
                filter_date_start = filter_date_end - dt.timedelta(days=days)
            else:
                filter_date_start = self.filing_date_objects.iloc[i - 1]
                filter_date_end = self.filing_date_objects.iloc[i]

            period_data = df[
                (df["date"] >= filter_date_start) & (df["date"] < filter_date_end)
            ]

            try:
                max_price = max(period_data["High"])
                min_price = min(period_data["Low"])
                avg_close = period_data["Close"].mean()
            except ValueError:
                working_array.append([np.nan] * 3)
            else:
                working_array.append([max_price, min_price, avg_close])

        cols = ["high", "low", "avg_close"]
        periodised = pd.DataFrame(
            working_array, index=self.frame_indecies, columns=cols
        )
        assert (
            sum(periodised["high"] < periodised["low"]) <= 1
        ), "Stock highs and lows not consistent"
        return periodised

    def save_financial_attributes(self):
        """
        Saves the financial data to local parquet files.
        """
        save_path = (
            Path.cwd().parent
            / "data"
            / "Company_Financial_Data"
            / self.ticker
            / self.period
        )
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print(f"Path already exists. Overwriting saved data.")
        except Exception:
            msg = f"Could not create directory {save_path}"
            raise Exception(msg)

        self.balance_sheets.to_parquet(save_path / "balance_sheets.parquet")
        self.income_statements.to_parquet(save_path / "income_statements.parquet")
        self.cash_flow_statements.to_parquet(save_path / "cash_flow_statements.parquet")
        self.stock_price_data.to_parquet(save_path / "stock_price_data.parquet")
        self.reported_key_metrics.to_parquet(save_path / "reported_key_metrics.parquet")


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
        self.assert_non_null_frame(self.calculated_metrics)
        self.calculation_error_dict = {}
        if self.verbose:
            self.fractional_metric_errors = self.cross_check_metric_calculations()
            self.assert_non_null_frame(self.fractional_metric_errors)
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
                print(f"{metric} series has sunsupported type for sum()")
        return fractional_errors


class Plots:
    """
    A class to create trend plots and export them as a PDF.

    Attributes:
        ticker (str): The stock ticker of the company to analyse.
        period (str): The period of data to analyse.
        metrics (pd.DataFrame): A DataFrame containing financial metrics to plot.
        limit (int): The maximum number of metrics to plot.
        filing_dates (list): A list of dates associated with the financial filings for the company.

    Methods:
        set_limit(limit: int) -> int:
            Set the limit to the maximum number of metrics to plot.
        get_spacing() -> int:
            Get the spacing for the x-axis labels.
        filter_x_labels(x_labels, spacing) -> list:
            Filter the x-axis labels to maintain a clean plot.
        generate_x_labels() -> list:
            Generate the x-axis labels for the plot.
        calculate_subplots_shape(metrics_container) -> Tuple[int, int]:
            Calculate the subplot shape for each metric.
        select_subplot(counter: int, subplots: plt.subplots) -> plt.subplot:
            Select the appropriate subplot.
        get_y_data(metric) -> pd.Series:
            Get the y-axis data for the plot.
        get_y_units(metric_type, metric) -> str:
            Get the units for the metric to label the y-axis.
        plot_data_on_axis(axis, data) -> plt.axis:
            Plot the data on the specified axis.
        get_linear_coeffs(x, y) -> Tuple[float, float, float]:
            Calculate the slope, intercept, and R-squared value for the linear trendline.
        generate_linear_series(x, slope, intercept) -> np.ndarray:
            Generate the linear trendline series.
        plot_linear_trend(axis, x, y, r2) -> plt.axis:
            Plot the linear trendline and R-squared value.
        plot_metrics():
            Plot the financial metrics and trendlines.
        generate_save_path_object(file=False) -> Path:
            Generate the file path object for the exported PDF.
        create_path_from_object(path_object):
            Create the path specified in the path object.
        make_bin_folder():
            Create a bin folder to store temporary files.
        make_pdf_title_page():
            Create the title page of the PDF.
        make_pdf_charts():
            Create the trend plot pages of the PDF.
        combine_and_export_pdfs(export_path):
            Combine the title page and trend plot pages and export the PDF.
        _export_charts_pdf():
            Export the trend plots as a PDF.

    """

    def __init__(self, ticker, period, metrics, limit, filing_dates):
        """
        Creates an instance of the `Plots` class.

        Args:
            ticker (str): The ticker symbol of the company.
            period (str): The reporting period of the financial data.
            metrics (pd.DataFrame): The financial metrics of the company.
            limit (int): The maximum number of metrics to display.
            filing_dates (List[str]): The filing dates of the financial data.

        Attributes:
            ticker (str): The ticker symbol of the company.
            period (str): The reporting period of the financial data.
            metrics (pd.DataFrame): The financial metrics of the company.
            filing_dates (List[str]): The filing dates of the financial data.
            limit (int): The maximum number of metrics to display.
            metric_units_dict (dict): A dictionary containing the units for each metric.
            plots (List[matplotlib.figure.Figure]): A list containing the plotted charts.
        """
        self.ticker = ticker
        self.period = period
        self.metrics = metrics
        self.filing_dates = filing_dates
        self.limit = self.set_limit(limit)
        self.metric_units_dict = {
            "Stock Evaluation Ratios": {
                "eps": "$/share",
                "eps_diluted": "$/share",
                "PE_high": "x",
                "PE_low": "x",
                "bookValuePerShare": "$/share",
                "dividendPayoutRatio": "x",
                "cashPerShare": "$/share",
                "ebitdaratio": "x",
            },
            "Profitability Ratios": {
                "grossProfitMargin": "x",
                "operatingProfitMargin": "x",
                "pretaxProfitMargin": "x",
                "netProfitMargin": "x",
                "ROIC": "x",
                "returnOnEquity": "x",
                "returnOnAssets": "x",
            },
            "Debt & Interest Ratios": {
                "interestCoverage": "x",
                "fixedChargeCoverage": "x",
                "debtToTotalCap": "x",
                "totalDebtRatio": "x",
            },
            "Liquidity Ratios": {
                "currentRatio": "x",
                "quickRatio": "x",
                "cashRatio": "x",
            },
            "Efficiency Ratios": {
                "totalAssetTurnover": "YYY",
                "inventoryToSalesRatio": "x",
                "inventoryTurnoverRatio": "YYY",
                "inventoryTurnoverInDays": "YYY",
                "accountsReceivableToSalesRatio": "YYY",
                "receivablesTurnover": "YYY",
                "receivablesTurnoverInDays": "YYY",
            },
        }

        self.plots = []
        self.plot_metrics()

    def set_limit(self, limit: int) -> int:
        """
        Sets the maximum number of metrics to display in the plots.

        Args:
            limit (int): The maximum number of metrics to display.

        Returns:
            int: The maximum number of metrics to display.
        """
        if limit > len(self.metrics):
            return len(self.metrics)
        else:
            return limit

    def get_spacing(self) -> int:
        """
        Determines the spacing between x-tick labels for the plots, based on the number of metrics plotted.

        Returns:
            int: The spacing between x-tick labels.
        """
        if self.limit < 10:
            return 2
        elif self.limit < 20:
            return 4
        else:
            return 6

    def filter_x_labels(self, x_labels, spacing):
        """
        Filters the x-tick labels for the plots, based on the specified spacing.

        Args:
            x_labels (list): The list of x-tick labels to be filtered.
            spacing (int): The spacing between x-tick labels.

        Returns:
            list: The filtered list of x-tick labels.
        """
        return [x_labels[i] if i % spacing == 0 else " " for i in range(len(x_labels))]

    def generate_x_labels(self):
        """
        Generates the x-tick labels for the plots, based on the metrics data and the specified period.

        Returns:
            list: The list of x-tick labels for the plots.
        """
        return [
            "-".join(str(i).split("-")[1:]) for i in self.metrics.index[-self.limit :]
        ]

    def calculate_subplots_shape(self, metrics_container):
        """
        Calculates the number of rows and columns required for the subplot grid, based on the number of 
            metrics to be plotted.

        Args:
            metrics_container (list): The list of metrics to be plotted.

        Returns:
            tuple: A tuple containing the number of rows and columns required for the subplot grid.
        """
        nplots = len(metrics_container)
        nrows = -(-nplots // 2)
        ncols = 2
        return nrows, ncols

    def select_subplot(self, counter: int, subplots: plt.subplots) -> plt.subplot:
        """
        Selects a subplot from the grid of subplots, based on the current counter and the subplots grid.

        Args:
            counter (int): The current counter value.
            subplots (matplotlib.figure.Subplot): The grid of subplots.

        Returns:
            plt.subplot: The selected subplot.
        """
        i, j = counter // 2, counter % 2
        return subplots[i][j]

    def get_y_data(self, metric):
        """
        Extracts the y-axis data for the specified metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            pd.Series: The y-axis data for the specified metric.
        """
        return self.metrics[metric][-self.limit :]

    def get_y_units(self, metric_type, metric):
        """
        Extracts the unit of measure for the y-axis for the specified metric.

        Args:
            metric_type (str): The category of the metric.
            metric (str): The name of the metric.

        Returns:
            str: The unit of measure for the y-axis for the specified metric.
        """
        return self.metric_units_dict[metric_type][metric]

    def plot_data_on_axis(self, axis, data):
        """
        Plots the data trend for the specified metric on the specified axis.

        Args:
            axis (plt.subplot): The axis on which to plot the data.
            data (dict): A dictionary containing the plotting data, including the metric data, labels, 
                and units of measure.
        """
        axis.plot(data["y"], label="data")
        axis.set_title(data["metric"])
        axis.set_xticks(data["x_true"])
        axis.set_xticklabels(data["x_labels"])
        axis.set_ylabel(["y_label"])
        return axis

    def get_linear_coeffs(self, x, y):
        """
        Calculate the linear regression coefficients and coefficient of determination (R-squared) 
        for the given x and y data points.

        Args:
            x (List[float]): List of x data points.
            y (List[float]): List of y data points.

        Returns:
            Tuple[float, float, float]: A tuple containing the slope, intercept, and R-squared value.

        Calculates the linear regression coefficients for the given x and y data points using the 
        `linregress` function from the `scipy.stats` module. The slope, intercept, and R-squared 
        value are returned as a tuple.
        """
        slope, intercept, r_value, _, _ = linregress(x, y)
        return slope, intercept, r_value ** 2

    def generate_linear_series(self, x, slope, intercept):
        """
        Returns a pandas Series containing the y-values of a linear series with the given x-values, 
        slope, and intercept.

        Args:
            x (pandas.Series): The x-values of the linear series.
            slope (float): The slope of the linear series.
            intercept (float): The intercept of the linear series.

        Returns:
            pandas.Series: The y-values of the linear series.
        """
        return slope * x + intercept

    def plot_linear_trend(self, axis, x, y, r2):
        """
        Plots linear regression line on given axis with r-squared value in legend.

        Args:
            axis (matplotlib axis): axis object on which the line will be plotted.
            x (np.ndarray): x-values for the data.
            y (np.ndarray): y-values for the data.
            r2 (float): r-squared value for the linear regression.

        Returns:
            matplotlib axis: axis object with the plotted line.
        """
        axis.plot(x, y, alpha=0.5, linestyle="--", label="linear trend")
        axis.plot([], [], " ", label=f"R2: {r2:.2f}")  # Adding R2 value to legend
        return axis

    def plot_metrics(self):
        """
        The plot_metrics method generates plots of financial metrics for a given company. The method first 
            generates the necessary subplots to hold each metric type. Each metric type is then plotted on 
            an individual axis within the metric type's subplot. Each metric's trend data is gathered, and 
            its trend is plotted on the selected axis. A linear trendline is also plotted along with its 
            corresponding R2 value. Finally, each metric type's subplot is given a title and added to the 
            list of plots.

        Args:
        self (Plots): The Plots instance.

        Returns:
        None. The method appends the generated plots to the Plots instance.
        """
        spacing = self.get_spacing()
        x_labels = self.filter_x_labels(self.generate_x_labels(), spacing)

        for metric_type in self.metric_units_dict.keys():
            # creating subplot axes to hold all metrics of each metric type
            metrics = self.metric_units_dict[metric_type].keys()
            nrows, ncols = self.calculate_subplots_shape(metrics)
            fig, ax = plt.subplots(nrows, ncols, figsize=(11.7, 8.3))

            # plotting each metric trend on an axis
            for counter, metric in enumerate(metrics):
                # axis selection
                axis = self.select_subplot(counter, ax)
                # gathering data trend for each metric
                y = self.get_y_data(metric)
                y_label = self.get_y_units(metric_type, metric)
                x_true = y.index
                x_dummy = range(len(y))

                # plotting the data
                plotting_data = {
                    "metric": metric,
                    "y": y,
                    "l_label": y_label,
                    "x_true": x_true,
                    "x_dummy": x_dummy,
                    "x_labels": x_labels,
                }
                self.plot_data_on_axis(axis, plotting_data)

                # plotting the linear trendline and R2
                slope, intercept, r2_value = self.get_linear_coeffs(x_dummy, y)
                y_linear = self.generate_linear_series(x_dummy, slope, intercept)
                self.plot_linear_trend(axis, x_dummy, y_linear, r2_value)
                axis.legend(loc="upper right", frameon=False, fontsize=8)

            # formatting and appending the figure
            fig.suptitle(metric_type)
            fig.tight_layout()
            self.plots.append(fig)

    def generate_save_path_object(self, file=False):
        """
        Generates the path to the PDF file where the charts will be saved.

        Args:
            file (bool, optional): Determines whether or not the filename should be appended to the file path. 
                Defaults to False.

        Returns:
            PosixPath: The file path object.
        """
        date = str(dt.datetime.now()).split()[0]
        end_date = self.filing_dates[-1]
        start_date = self.filing_dates[-self.limit]
        file_name = (
            f"{self.ticker}_{self.period}_{str(start_date)}_to_{str(end_date)}.pdf"
        )
        file_path = (
            Path.cwd()
            /"investment_tools"
            / "data"
            / "Company Analysis"
            / date
            / self.ticker
            / self.period
        )
        return file_path / file_name if file else file_path

    def create_path_from_object(self, path_object):
        """
        Creates a directory at the given path if it does not already exist.

        Args:
            path_object (Path): The path to create the directory.
        """
        try:
            os.makedirs(path_object)
        except FileExistsError:
            pass

    def make_bin_folder(self):
        """
        Creates a 'bin' folder if it does not already exist.
        """
        try:
            os.mkdir("bin")
        except FileExistsError:
            pass

    def make_pdf_title_page(self):
        """
        Generates a PDF title page using the ticker and period of the company, and saves it in the 'bin' folder.
        """
        self.title_path = "bin/title.pdf"
        self.charts_path = "bin/charts.pdf"
        title_message = f"Financial Ratio Trends for {self.ticker}"
        title_page = canvas.Canvas(self.title_path)
        title_page.drawString(210, 520, title_message)
        title_page.save()

    def make_pdf_charts(self):
        """
        Generates individual PDFs for each figure in self.plots, and saves them in the 'bin' folder.
        """
        self.charts_path = "bin/charts.pdf"
        with PdfPages(self.charts_path) as pdf:
            for figure in self.plots:
                pdf.savefig(figure)

    def combine_and_export_pdfs(self, export_path):
        """
        Combines the PDFs generated in make_pdf_title_page and make_pdf_charts, and exports them to the given path.

        Args:
            export_path (str): The file path to export the combined PDF to.
        """
        with open(self.title_path, "rb") as f1:
            with open(self.charts_path, "rb") as f2:
                pdf1 = PdfReader(f1, "rb")
                pdf2 = PdfReader(f2, "rb")
                pdf_output = PdfWriter()
                for page_num in range(len(pdf1.pages)):
                    pdf_output.add_page(pdf1.pages[page_num])
                for page_num in range(len(pdf2.pages)):
                    pdf_output.add_page(pdf2.pages[page_num])
                with open(export_path, "wb") as output_file:
                    pdf_output.write(output_file)

    def _export_charts_pdf(self):
        """
        Generates and exports a combined PDF file of the financial ratio trend plots to a file path
        specified by generate_save_path_object.
        """
        path_object = self.generate_save_path_object(file=False)
        self.create_path_from_object(path_object)
        self.make_bin_folder()
        self.make_pdf_title_page()
        self.make_pdf_charts()
        path_object = self.generate_save_path_object(file=True)
        self.combine_and_export_pdfs(path_object)


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
        self, ticker, api_key, data="online", period="annual", limit=20, verbose=False
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
            # self._plots = Plots(self.ticker, self.period, self.metrics, limit, self.filing_dates)
            # self.trends = self._plots.plots
        self.scores = self.recommendation()
        self.outcome = self.eval_(self.scores)
        if self.outcome:
            self.print_charts()
            self.export()

    def recommendation(self) -> Dict[str, dict]:
        """
        Compute a recommendation for the company based on key financial metrics.

        Returns:
        scores (Dict[str, dict]): A dictionary containing the scores and strengths for each 
        financial metric considered.
        """
        key_metrics = {
            "eps": "eps_growth",
            "returnOnEquity": "returnOnEquity_growth",
            "ROIC": "ROIC_growth",
            "returnOnAssets": "returnOnAssets_growth",
            "debtToTotalCap": "debtToTotalCap_growth",
            "totalDebtRatio": "totalDebtRatio_growth"
        }
        scores = dict()
        for metric, growth in key_metrics.items():
            score, strength = self.score(metric, growth)
            scores[metric] = {"score": score, "strength": strength}
        return scores

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

    def score(self, metric: str, growth: str) -> Tuple[int, int]:
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
        growth_ = self.get_copy_of_df_column(growth)
        mean_growth = growth_.mean()
        r2 = self.get_r2_val(metric_)
        growth_score = self.score_mean_growth(modifier * mean_growth)
        stability_score = self.score_trend_strength(r2)
        return (growth_score, stability_score)

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
        if growth <= 0.05:
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
        if r2_ <= 0.2:
            return 0
        elif r2_ <= 0.3:
            return 1
        elif r2_ <= 0.5:
            return 2
        elif r2_ <= 0.75:
            return 3
        else:
            return 4

    def eval_(self, scores: Dict[str, dict]) -> bool:
        """
        Determines if the company has favorable financial metrics based on the provided scores.

        Args:
        scores (Dict[str, dict]): A dictionary of scores and strengths for various financial metrics

        Returns:
        bool: True if the company has favorable financial metrics based on the provided scores, False otherwise.
        
        """
        vote = 0
        for key in scores.keys():
            vote += scores[key]["score"]
        # Require an average score of 1.5 for each metric
        return False if vote < 1.5 * len(scores) else True

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
