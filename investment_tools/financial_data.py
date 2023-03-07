import datetime as dt
import yfinance as yf
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import requests
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Tuple, List
import pyarrow as pa

yf.pdr_override()


class FinancialData:
    """
    A class for retrieving and storing financial data for a given stock ticker.

    Attributes:
        ticker (str): The stock ticker symbol.
        api_key (str, optional): An optional API key to access financial data.
        data (str): Specifies whether to retrieve data from an online source or a local data store.
        period (str): The time period for which data is retrieved, either "annual" or "quarterly".
        limit (int): The maximum number of financial statements to retrieve.
        balance_sheets (pandas.DataFrame): The balance sheet data for the stock.
        income_statements (pandas.DataFrame): The income statement data for the stock.
        cash_flow_statements (pandas.DataFrame): The cash flow statement data for the stock.
        reported_key_metrics (pandas.DataFrame): The reported key metric data for the stock.
        stock_price_data (pandas.DataFrame): The stock price data for the stock.

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
        data: str="local",
        period: str="annual",
        limit: int=120,
    ):
        """
        Initializes a new instance of the FinancialData class.

        Args:
            ticker (str): The stock ticker symbol.
            api_key (str, optional): An optional API key to access financial data.
            data (str): Specifies whether to retrieve data from an online source or a local data store.
            period (str): The time period for which data is retrieved, either "annual" or "quarterly".
            limit (int): The maximum number of financial statements to retrieve.
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
        """
        Replaces None values in a DataFrame with 0.

        Args:
            df (pandas.DataFrame): The DataFrame to process.

        Returns:
            pandas.DataFrame: The processed DataFrame with None values replaced by 0.
        """
        df_ = df.replace(to_replace=np.nan, value=0)
        return df_.replace(to_replace=[None,'None'], value=0)

    def assert_valid_user_inputs(self) -> None:
        """
        Asserts that the user inputs for the FinancialData instance are valid.

        Raises:
            AssertionError: If the data or period attributes are invalid.
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
            data_type (str): The type of financial data to fetch. Acceptable values are 'bs', 'is', 'cfs', and 'ratios'.
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
            Path.cwd()
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
        Set the index of another dataframe to be equal to the index of the balance sheet dataframe.

        Args:
        other (pd.DataFrame): A dataframe whose index is to be set to that of the balance sheet dataframe.

        Returns:
        pd.DataFrame: A copy of the other dataframe with its index set to that of the balance sheet dataframe.
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

    def filter_for_common_indecies(self, common_elements: pd.Index) -> None:
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

    def assert_identical_indecies(self) -> None:
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

    def assert_required_length(self, item: list) -> None:
        """
        Asserts that a given item has the required length based on the period.
        
        Args:
            item (pandas.DataFrame): The item to check the length of.
        """
        if self.period == "annual":
            required_length = 3
        else:
            required_length = 10
        err_msg = f"Financial statements are shorter than the required length of {required_length}"
        assert len(item) >= required_length, err_msg

    def assert_valid_server_response(self, response: requests.Response) -> None:
        """
        Asserts that the server response is valid (status code 200).
        
        Args:
            response (requests.Response): The server response.
        """
        assert (
            response.status_code == 200
        ), f"API call failed. Code <{response.status_code}>"

    def assert_server_response_not_empty(self, response: requests.Response) -> None:
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
        if raw_data.empty:
            return self.generate_empty_df(["high", "low", "avg_close"])
        raw_data["date"] = raw_data.index.date
        return self.periodise(raw_data)

    def generate_empty_df(self, columns: pd.Index) -> pd.DataFrame:
        """
        Generate an empty dataframe with the global frame indices and provided column names.
    
        Args:
            columns (pandas.Index): the provided columns.

        Returns:
            pandas.DataFrame: the empty dataframe.
        """
        data = [[0]*len(columns)]*len(self.balance_sheets)
        index = self.balance_sheets.index
        return pd.DataFrame(data, index=index, columns=columns)

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

    def save_financial_attributes(self) -> None:
        """
        Saves the financial data to local parquet files.
        """
        save_path = (
            Path.cwd()
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

        try:
            self.balance_sheets.to_parquet(save_path / "balance_sheets.parquet")
            self.income_statements.to_parquet(save_path / "income_statements.parquet")
            self.cash_flow_statements.to_parquet(save_path / "cash_flow_statements.parquet")
            self.stock_price_data.to_parquet(save_path / "stock_price_data.parquet")
            self.reported_key_metrics.to_parquet(save_path / "reported_key_metrics.parquet")
        except pa.ArrowTypeError as e:
            print(f"financial data could not be saved.")
            raise e