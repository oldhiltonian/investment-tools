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

yf.pdr_override()

class FinancialData:
    """
    A class for fetching and processing financial data for a given stock.

    Attributes:
        ticker (str): The ticker symbol for the stock.
        api_key (str): The user's API key for accessing financial data (optional).
        data (str): Whether to fetch data from online or local source.
        period (str): The period of the data, either 'annual' or 'quarter'.
        limit (int): The maximum number of data points to fetch from the API.

    Raises:
        AssertionError: If invalid input is provided for data or period.

    """
    def __init__(self, ticker, api_key='', data='local', period='annual', limit=120):
        """
        Initializes a new instance of the FinancialData class.

        Args:
            ticker (str): The ticker symbol for the stock.
            api_key (str): The user's API key for accessing financial data (optional).
            data (str): Whether to fetch data from online or local source.
            period (str): The period of the data, either 'annual' or 'quarter'.
            limit (int): The maximum number of data points to fetch from the API.

        """
        self.ticker = ticker.upper().strip()
        self.api_key = str(api_key)
        self.data = data.lower().strip()
        self.period = period.lower().strip()
        self.assert_valid_user_inputs()
        self.limit = int(limit)
        self.days_in_period = 365 if period == 'annual' else 90

        self.balance_sheets       = self.build_dataframe(self.fetch_raw_data('bs').json())
        self.income_statements    = self.build_dataframe(self.fetch_raw_data('is').json())
        self.cash_flow_statements = self.build_dataframe(self.fetch_raw_data('cfs').json())
        self.reported_key_metrics = self.build_dataframe(self.fetch_raw_data('metrics').json())

        if not self.check_for_matching_indecies():
            self.filter_for_common_indecies(self.get_common_df_indicies())
        
        self.assert_required_length(self.balance_sheets)
        self.frame_indecies = self.get_frame_indecies()
        self.filing_date_objects = self.balance_sheets['date']
        self.stock_price_data = self.fetch_stock_price_data_yf()
        self.assert_required_length(self.stock_price_data)
        self.save_financial_attributes()
        

    def assert_valid_user_inputs(self):
        """
        Ensures that valid user inputs are provided for data and period.
        
        Raises:
            AssertionError: If invalid input is provided for data or period.
        """
        assert self.data in ['online', 'local'], "data must be 'online' or 'local'"
        assert self.period in ['annual', 'quarter'], "period must be 'annual' or 'quarter"


    def generate_request_url(self, data_type):
        """
        Generates a URL to fetch data from the Financial Modeling Prep API.
        
        Args:
            data_type (str): The type of data to fetch.
        
        Returns:
            str: The URL for the requested data.
        
        Raises:
            ValueError: If an invalid data type is provided.
        """
        fmp_template = 'https://financialmodelingprep.com/api/v3/{}/{}?period={}&limit={}&apikey={}'
        ticker = self.ticker
        period = self.period
        limit = self.limit
        api_key = self.api_key

        if data_type == 'bs':
            data_str = 'balance-sheet-statement'
        elif data_type == 'is':
            data_str = 'income-statement'
        elif data_type == 'cfs':
            data_str = 'cash-flow-statement'
        elif data_type == 'metrics':
            data_str = 'ratios'
        else:
            err_msg = f"{data_type} is not a valid API call"
            raise ValueError(err_msg)
        
        return fmp_template.format(data_str, ticker, period, limit, api_key)
    

    def fetch_raw_data(self, data_type):
        """
        Fetches raw financial data from either the Financial Modeling Prep API or a local file.
        
        Args:
            data_type (str): The type of data to fetch.
        
        Returns:
            pandas.DataFrame: The fetched financial data.
        """
        if self.data == 'online':
            url = self.generate_request_url(data_type)
            raw_data = requests.get(url)
            self.assert_valid_server_response(raw_data)
        elif self.data == 'local':
            path = self.get_load_path(data_type, self.ticker, self.period)
            raw_data = pd.read_parquet(path)
        return raw_data
            

    def get_load_path(self, data_type, ticker, period):
        """
        Gets the file path to load a local financial data file.
        
        Args:
            data_type (str): The type of financial data to fetch.
            ticker (str): The ticker symbol for the stock.
            period (str): The period of the data, either 'annual' or 'quarter'.
        
        Returns:
            pathlib.Path: The file path for the requested data.
        """
        return Path.cwd()/'Company Financial Data'/ticker/period/data_type/'.parquet'


    def get_frame_indecies(self):
        """
        Gets the index for the financial data.
        
        Returns:
            pandas.Index: The index for the financial data.
        """
        self.frame_indecies = self.balance_sheets.index


    def build_dataframe(self, data):
        """
        Builds a pandas.DataFrame from provided raw data.
        
        Args:
            data (dict): The raw data to build the DataFrame from.
        
        Returns:
            pandas.DataFrame: The built DataFrame.
        """
        working_array = []
        for statement in reversed(data):
            working_array.append(list(statement.values()))
        df = pd.DataFrame(working_array, columns = data[0].keys())
        df['index'] = df['date'].apply(lambda x: self.generate_index(x))
        df['date'] = df['date'].apply(lambda x: self.generate_date(x))
        df = df.set_index('index')
        if 'netIncome' and 'eps' in df.keys():
            df['outstandingShares_calc'] = df['netIncome']/df['eps']
        return df


    def generate_index(self, date):
        """
        Generates an index for the financial data.
        
        Args:
            date (str): The date in 'YYYY-MM-DD' format.
        
        Returns:
            str: The generated index string.
        """
        year, month, _ = [int(i) for i in date.split('-')]
        
        if self.period == 'annual':
            return f"{self.ticker}-FY-{year}"

        if month in (1,2,3):
            quarter = 1
        elif month in (4,5,6):
            quarter = 2
        elif month in (7,8,9):
            quarter = 3
        elif month in (10, 11, 12):
            quarter = 4
        
        return f"{self.ticker}-Q{quarter}-{year}"


    def generate_date(self, date_str):
        """
        Generates a datetime.date object from a date string.
        
        Args:
            date_str (str): The date in 'YYYY-MM-DD' format.
        
        Returns:
            datetime.date: The generated date object.
        """
        year, month, day = [int(i) for i in date_str.split()[0].split('-')]
        return dt.date(year, month, day)


    def check_for_matching_indecies(self):
        """
        Checks whether the financial data has matching indices.
        
        Returns:
            bool: True if the financial data has matching indices, False otherwise.
        """
        print('Checking matching frame indecies')
        len_bs = len(self.balance_sheets)
        len_is = len(self.income_statements)
        len_cfs = len(self.cash_flow_statements)
        len_ratios = len(self.reported_key_metrics)
        print(f"Financial statement lengths are BS: {len_bs}, IS:{len_is}, CFS:{len_cfs}, Ratios:{len_ratios}")
        # Logical checks for matching indecies
        matching_index_1 =  self.balance_sheets['date'].equals(self.income_statements['date'])
        matching_index_2 = self.balance_sheets['date'].equals(self.cash_flow_statements['date'])
        matching_index_3 = self.balance_sheets['date'].equals(self.reported_key_metrics['date'])
        matching_indecies = matching_index_1 and matching_index_2 and matching_index_3
        return True if matching_indecies else False


    def get_common_df_indicies(self):
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


    def filter_for_common_indecies(self, common_elements):
        """
        Filters the financial data to only include common indices.
        
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
        Asserts that the financial data has identical indices for each of its statements.
        """
        assert len(self.cash_flow_statements) == len(self.balance_sheets), 'Indecies could not be filtered for common elements'
        assert len(self.income_statements) == len(self.balance_sheets), 'Indecies could not be filtered for common elements'
        assert len(self.reported_key_metrics) == len(self.balance_sheets), 'Indecies could not be filtered for common elements'


    def assert_required_length(self, item):
        """
        Asserts that a given item has the required length based on the period.
        
        Args:
            item (pandas.DataFrame): The item to check the length of.
        """
        if self.period == 'annual':
            required_length = 2
        else:
            required_length = 4
        err_msg = f"Financial statements are shorter than the required length of {required_length}"
        assert len(item) >= required_length, err_msg

    def assert_valid_server_response(Self, response):
        """
        Asserts that the server response is valid (status code 200).
        
        Args:
            response (requests.Response): The server response.
        """
        assert response.status_code == 200, f"API call failed. Code <{response.status_code}>"


    def fetch_stock_price_data_yf(self):
        """
        Fetches stock price data from Yahoo Finance.
        
        Returns:
            pandas.DataFrame: The fetched stock price data.
        """
        start_date = dt.date(*[int(i) for i in str(self.filing_date_objects.iloc[0]).split()[0].split('-')]) - dt.timedelta(days=370)
        end_date = dt.date.today()
        price_interval = '1d'
        raw_data = pdr.get_data_yahoo(self.ticker, start_date, end_date, interval=price_interval)
        raw_data['date'] = raw_data.index.date
        return self.periodise(raw_data)


    def periodise(self, df):
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
                filter_date_start = self.filing_date_objects.iloc[i-1]
                filter_date_end = self.filing_date_objects.iloc[i]

            period_data = df[(df['date'] >= filter_date_start) & (df['date'] < filter_date_end)]

            try:
                max_price = max(period_data['High'])
                min_price = min(period_data['Low'])
                avg_close = period_data['Close'].mean()
            except ValueError:
                working_array.append([np.nan]*3)
            else:
                working_array.append([max_price, min_price, avg_close])

        cols = ['high', 'low', 'avg_close']
        periodised = pd.DataFrame(working_array, index=self.frame_indecies, columns=cols)
        assert sum(periodised['high'] < periodised['low']) <= 1, 'Stock highs and lows not consistent'
        return periodised


    def save_financial_attributes(self):
        """
        Saves the financial data to local parquet files.
        """
        save_path = Path.cwd()/'investment_tools'/'data'/'Company Financial Data'/self.ticker/self.period
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print(f"Path already exists. Overwriting saved data.")
        except Exception:
            msg = f'Could not create directory {save_path}'
            raise Exception(msg)
        
        self.balance_sheets.to_parquet(save_path/'balance_sheets.parquet')
        self.income_statements.to_parquet(save_path/'income_statements.parquet')
        self.cash_flow_statements.to_parquet(save_path/'cash_flow_statements.parquet')
        self.stock_price_data.to_parquet(save_path/'stock_price_data.parquet')
        self.reported_key_metrics.to_parquet(save_path/'reported_key_metrics.parquet')





class ManualAnalysis:
    """Financial Data Analysis class

    Analyzes financial data and returns important metrics and ratios.
    
    Parameters
    ----------
    financial_data : object
        Financial data to be analyzed
    
    Attributes
    ----------
    data : object
        Financial data
    calculated_metrics : pd.DataFrame
        Metrics calculated from the financial data
    reported_metrics : pd.DataFrame
        Metrics reported in the financial data
    metric_errors : pd.DataFrame
        Error between calculated and reported metrics
    ratio_errors : pd.DataFrame
        Error between calculated and reported ratios
    metrics : dict
        Dictionary of financial metrics and ratios
    """
    def __init__(self, financial_data, verbose=False):
        super().__init__()
        self.data = financial_data
        self.verbose = verbose
        clc, rep, met_err, rat_err = self.cross_check_statement_calculations()
        self.calculated_statement_metrics = clc
        self.reported_statement_metrics= rep
        self.statement_metric_errors = met_err
        self.statement_ratio_errors = rat_err
        self.statement_metrics = self.analyse()
        self.cross_check_metric_calculations()

    
    def print_metric_errors(self, metric_errors, tolerance=0.05):
        if not self.verbose:
            return
        self.error_dict = dict()
        line_count = len(metric_errors)
        for metric in metric_errors:
            if metric is not None:
                count = sum(metric_errors[metric] >= tolerance)
                message = f"There were {count}/{line_count} values in {metric} that exceed the {tolerance} error tolerance."
                self.error_dict[metric] = (count, message)
        for tup in self.error_dict.values():
            print(tup[1])

    def cross_check_statement_calculations(self):
        """
        Calculates financial metrics and and compares them to the reported values.
        
        Returns pandas DataFrames representing the calculated values, reported values,
        the error between the calculated and reported values, and the error of just
        the financial ratios"""
    
        reported = pd.DataFrame(index=self.data.frame_indecies)
        calculated = pd.DataFrame(index=self.data.frame_indecies)
        RND_expenses = self.data.income_statements['researchAndDevelopmentExpenses']
        SGA_expenses = self.data.income_statements['sellingGeneralAndAdministrativeExpenses']
        other_expenses = self.data.income_statements['otherExpenses']
        revenue = self.data.income_statements['revenue']
        self.revenue = revenue
        cost_of_revenue = self.data.income_statements['costOfRevenue']
        depreciation_amortization = self.data.cash_flow_statements['depreciationAndAmortization']
        interest_expense = self.data.income_statements['interestExpense']
        interest_income = self.data.income_statements['interestIncome']
        net_income_reported = self.data.income_statements['netIncome']
        outstanding_shares = self.data.income_statements['outstandingShares_calc']



        metrics = ['ebitda', 'ebitdaratio', 'grossProfit', 'grossProfitRatio', 'operatingIncome', 'operatingIncomeRatio', \
                    'incomeBeforeTax', 'incomeBeforeTaxRatio', 'netIncome', 'netIncomeRatio', 'eps']

        # Calculated ratios from reported values on the financial statements
        calculated['ebitda'] = revenue - cost_of_revenue- RND_expenses - SGA_expenses - other_expenses + depreciation_amortization
        calculated['ebitdaratio'] = calculated['ebitda']/revenue
        calculated['grossProfit'] = revenue - cost_of_revenue
        calculated['grossProfitRatio'] = (calculated['grossProfit']/revenue)
        calculated['operatingIncome'] = revenue - cost_of_revenue - SGA_expenses - RND_expenses
        calculated['operatingIncomeRatio'] = (calculated['operatingIncome']/revenue)
        calculated['incomeBeforeTax'] = calculated['operatingIncome'] - interest_expense + interest_income
        calculated['incomeBeforeTaxRatio'] = (calculated['incomeBeforeTax']/revenue)
        calculated['netIncome'] = 0.79*calculated['incomeBeforeTax']
        calculated['netIncomeRatio'] = (calculated['netIncome']/revenue)
        calculated['eps'] = net_income_reported/outstanding_shares
        
        # Pulling reported metric values
        for metric in metrics:
            if metric in self.data.income_statements.keys():
                reported[metric] = self.data.income_statements[metric]
            elif metric in self.balance_sheets.keys():
                reported[metric] = self.data.balance_sheets[metric]
            elif metric in self.data.cash_flow_statements.keys():
                reported[metric] = self.data.cash_flow_statements[metric]

        # Ensuring dimensionality of the two dataframes
        if len(calculated.keys()) != len(reported.keys()):
            msg = '''Key mismatch between the reported and calculated tables.\nCheck the calculations in the Company.cross_check() method'''
            raise Exception(msg)
        # Error between calculated and reported values
        metric_errors = calculated - reported
        ratio_errors = metric_errors.drop(['ebitda','grossProfit', 'operatingIncome', 'incomeBeforeTax', 'netIncome'], inplace=False, axis=1)
        self.print_metric_errors(ratio_errors)
        return calculated, reported, metric_errors, ratio_errors

    def analyse(self):
        """Calculates and returns important financial metrics and ratios as a 
            pandas DataFrame."""
        df = pd.DataFrame(index=self.data.frame_indecies)

        '''Stock Evaluation Ratios'''
        total_assets = self.data.balance_sheets['totalAssets']
        total_liabilities = self.data.balance_sheets['totalLiabilities']
        long_term_debt = self.data.balance_sheets['longTermDebt']
        dividends_paid = self.data.cash_flow_statements['dividendsPaid']
        outstanding_shares = self.data.income_statements['outstandingShares_calc']
        cash_and_equivalents = self.data.balance_sheets['cashAndCashEquivalents']
        eps = self.data.income_statements['eps']
        stock_price_high = self.data.stock_price_data['high']
        stock_price_avg = self.data.stock_price_data['avg_close']
        stock_price_low = self.data.stock_price_data['low']
        df['eps'] = eps #authorized stock!!!
        df['eps_diluted'] = self.data.income_statements['epsdiluted']
        df['PE_high'] = stock_price_high/eps
        df['PE_low'] = stock_price_low /eps
        df['PE_avg_close'] = stock_price_avg/eps
        df['bookValuePerShare'] = (total_assets-total_liabilities)/outstanding_shares
        df['dividendPayoutRatio'] = (-dividends_paid/outstanding_shares)/eps
        df['dividendYield_low'] = (-dividends_paid/outstanding_shares)/stock_price_high
        df['dividendYield_high'] = (-dividends_paid/outstanding_shares)/stock_price_low 
        df['dividendYield_avg_close'] = (-dividends_paid/outstanding_shares)/stock_price_avg
        df['ebitdaratio'] = self.data.income_statements['ebitdaratio']
        df['cashPerShare'] = (1*(cash_and_equivalents-long_term_debt))/outstanding_shares


        '''Profitability Ratios'''
        revenue = self.data.income_statements['revenue']
        gross_profit = self.data.income_statements['grossProfit']
        COGS = self.data.income_statements['costOfRevenue']
        SGA = self.data.income_statements['sellingGeneralAndAdministrativeExpenses']
        RND_expense = self.data.income_statements['researchAndDevelopmentExpenses']
        operating_income = self.data.income_statements['operatingIncome']
        income_before_tax = self.data.income_statements['incomeBeforeTax']
        total_capitalization = self.data.balance_sheets['totalEquity'] + self.data.balance_sheets['longTermDebt']
        net_income = self.data.income_statements['netIncome']
        total_shareholder_equity = self.data.balance_sheets['totalStockholdersEquity']
        df['grossProfitMargin'] = gross_profit/revenue
        df['operatingProfitMargin'] = operating_income/revenue
        df['pretaxProfitMargin'] = income_before_tax/revenue
        df['netProfitMargin'] = net_income/revenue
        df['ROIC'] = net_income/total_capitalization
        df['returnOnEquity'] = net_income/total_shareholder_equity
        df['returnOnAssets'] = net_income/total_assets

        '''Debt and Interest Ratios'''
        interest_expense = self.data.income_statements['interestExpense']
        # The fixed_charges calculation below is likely incomplete
        fixed_charges = self.data.income_statements['interestExpense'] + self.data.balance_sheets['capitalLeaseObligations']
        ebitda = self.data.income_statements['ebitda']
        total_equity = self.data.balance_sheets['totalEquity']
        total_debt = total_assets - total_equity
        df['interestCoverage'] = operating_income/interest_expense
        df['fixedChargeCoverage'] = ebitda/fixed_charges
        df['debtToTotalCap'] = long_term_debt/total_capitalization
        df['totalDebtRatio'] = total_debt/total_assets

        '''Liquidity & FinancialCondition Ratios'''
        current_assets = self.data.balance_sheets['totalCurrentAssets']
        current_liabilities = self.data.balance_sheets['totalCurrentLiabilities']
        inventory = self.data.balance_sheets['inventory']
        quick_assets = current_assets - inventory
        df['currentRatio'] = current_assets/current_liabilities
        df['quickRatio'] = quick_assets/current_liabilities
        df['cashRatio'] = cash_and_equivalents/current_liabilities


        '''Efficiency Ratios'''
        net_receivables = self.data.balance_sheets['netReceivables']
        df['totalAssetTurnover'] = revenue/total_assets
        df['inventoryToSalesRatio'] = inventory/revenue
        df['inventoryTurnoverRatio'] = 1/df['inventoryToSalesRatio']
        days = 365 if self.data.period == 'annual' else 90
        df['inventoryTurnoverInDays'] = days/df['inventoryTurnoverRatio']
        accounts_receivable_to_sales_ratio = self.data.balance_sheets['netReceivables']/revenue
        df['accountsReceivableToSalesRatio'] = accounts_receivable_to_sales_ratio
        df['receivablesTurnover'] = revenue/net_receivables
        df['receivablesTurnoverInDays'] = days/df['receivablesTurnover']


        '''Metric Growth'''
        metrics = ['eps', 'returnOnEquity', 'cashPerShare', 'PE_avg_close', 'ebitdaratio', 'ROIC',
                   'netProfitMargin', 'returnOnAssets', 'debtToTotalCap', 'totalDebtRatio']
        span, init_idx = (1,1) if self.data.period == 'annual' else (4,4)
        for metric in metrics:
            series = df[metric]
            col_header = metric+'_growth'
            col_data = []
            for i in range(len(series)):
                if i < init_idx:
                    col_data.append(np.nan)
                else:
                    col_data.append((series[i]-series[i-span])/abs(series[i-span]))
            df[col_header] = pd.Series(col_data, index=self.data.frame_indecies)
        return df

    def cross_check_metric_calculations(self):
        df = pd.DataFrame()
        metrics = ['grossProfitMargin', 'operatingProfitMargin', 'currentRatio', 
                    'returnOnEquity', 'returnOnAssets', 'cashPerShare', 'interestCoverage',
                    'dividendPayoutRatio']
        
        for metric in metrics:
            df[metric] = self.statement_metrics[metric] - self.data.reported_key_metrics[metric]
        self.print_metric_errors(df, 0.05)

class Plots:
    #self.n needs to be handled properly in the plot() function 
    """
    Class to handle the plotting of financial ratios.

    Parameters:
    data (DataFrame): The data used for plotting
    n (int): The number of data points used for plotting

    Attributes:
    ratio_dict (dict): The financial ratios to be plotted
    plots (list): List of all the plotted figures
    """
    '''
    
    '''
    def __init__(self, ticker, period, metrics, limit, filing_dates):
        self.ticker = ticker
        self.period = period
        self.data = metrics
        self.limit = limit
        self.filing_dates = filing_dates
        if limit > len(metrics):
            self.limit = len(metrics)
        else:
            self.limit = limit
        self.metric_units_dict = {
            'Stock Evaluation Ratios':  {'eps' : '$/share',
                                         'eps_diluted': '$/share',
                                         'PE_high': 'x',
                                         'PE_low': 'x',
                                         #'PE_avg_close': 'x',
                                         'bookValuePerShare': '$/share',
                                         'dividendPayoutRatio': 'x',
                                         #'dividendYield_avg_close': 'x',
                                         'cashPerShare': '$/share',
                                         'ebitdaratio': 'x'
                                        },
            'Profitability Ratios':     {'grossProfitMargin': 'x',
                                         'operatingProfitMargin': 'x',
                                         'pretaxProfitMargin': 'x',
                                         'netProfitMargin': 'x',
                                         'ROIC': 'x',
                                         'returnOnEquity': 'x',
                                         'returnOnAssets': 'x'
                                        },
            'Debt & Interest Ratios':   {'interestCoverage': 'x',
                                         'fixedChargeCoverage': 'x',
                                         'debtToTotalCap': 'x',
                                         'totalDebtRatio': 'x'
                                         },
            'Liquidity Ratios':         {'currentRatio': 'x',
                                         'quickRatio': 'x',
                                         'cashRatio': 'x'},
            'Efficiency Ratios':     {'totalAssetTurnover': 'YYY',
                                         'inventoryToSalesRatio': 'x',
                                         'inventoryTurnoverRatio': 'YYY',
                                         'inventoryTurnoverInDays': 'YYY',
                                         'accountsReceivableToSalesRatio': 'YYY',
                                         'receivablesTurnover': 'YYY',
                                         'receivablesTurnoverInDays': 'YYY'
                                         }
 }

        self.plots = []
        self.plot_metrics()

    def plot_metrics(self):
        """
        Plots the financial ratios using matplotlib.

        Returns:
        None

        ###delete
        ticker-Q2-2021 or FY
        """

        x_labels = ['-'.join(str(i).split('-')[1:]) for i in self.data.index[-self.limit:]]
        if self.limit < 10:
            spacing = 2
        elif self.limit < 20:
            spacing = 4
        else:
            spacing = 6
        
        for metric_type in self.metric_units_dict.keys():
            metrics_dict = self.metric_units_dict[metric_type]
            metrics = metrics_dict.keys()
            nplots = len(metrics_dict)
            nrows = -(-nplots//2)
            fig, ax = plt.subplots(nrows, 2, figsize=(11.7, 8.3))

            for counter, metric in enumerate(metrics):
                # targetting the right subplot
                i, j = counter//2, counter%2
                axis = ax[i][j]

                # plotting the actual metric values
                y = self.data[metric][-self.limit:]
                x = y.index
                x_dummy = range(len(y))
                
                axis.plot(y, label='data')
                axis.set_title(metric)
                axis.set_xticks(x)
                axis.set_xticklabels([x_labels[i] if i%spacing==0 else ' ' for i in range(len(x_labels))])
                y_label = self.metric_units_dict[metric_type][metric]
                axis.set_ylabel(y_label)

                # plotting the linear trendline and R2
                slope, intercept, r_value, _, _ = linregress(x_dummy, y)
                y_linear = slope*x_dummy + intercept
                axis.plot(x_dummy, y_linear, alpha=0.5, linestyle='--', label='linear trend')
                axis.plot([], [], ' ', label=f'R2: {r_value**2:.2f}') # Adding R2 value to legend
                axis.legend(loc='upper right', frameon=False, fontsize=8)

            # formatting and append
            fig.suptitle(metric_type)
            fig.tight_layout()
            self.plots.append(fig)
        
    def _export_charts_pdf(self):
        """
        Creates the financial trend charts as a pdf file.
        
        """
        date = str(dt.datetime.now()).split()[0]
        end_date = self.filing_dates[-1]
        start_date = self.filing_dates[-self.limit]
        file_name = f"{self.ticker}_{self.period}__{str(start_date)}_to_{str(end_date)}.pdf"
        file_path = Path.cwd()/'investment_tools'/'data'/'Company Analysis'/date/self.ticker/self.period

        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass
        
        try:
            os.mkdir('bin')
        except FileExistsError:
            pass

        # Making title page
        title_path = 'bin/title.pdf'
        charts_path = 'bin/charts.pdf'
        title_message = f"Financial Ratio Trends for {self.ticker}"
        title_page = canvas.Canvas(title_path)
        title_page.drawString(210, 520, title_message)
        title_page.save()



        with PdfPages(charts_path) as pdf:
            for figure in self.plots:
                pdf.savefig(figure)
    
        with open(title_path, 'rb') as f1:
            with open(charts_path, 'rb') as f2:
                pdf1 = PdfReader(f1, 'rb')
                pdf2 = PdfReader(f2, 'rb')
                pdf_output = PdfWriter()
                for page_num in range(len(pdf1.pages)):
                    pdf_output.add_page(pdf1.pages[page_num])
                for page_num in range(len(pdf2.pages)):
                    pdf_output.add_page(pdf2.pages[page_num])
                with open(file_path/file_name, 'wb') as output_file:
                    pdf_output.write(output_file)




class Company:
    """
    Class representing a company and its financial analysis

    Args:
    ticker (str): The stock symbol representing the company
    api_key (str): API key for accessing financial data from a data source
    data (str, optional): Source of data either from 'online' (default) or 'offline'
    period (str, optional): Financial period to use for the analysis either 'annual' (default) or 'quarterly'
    limit (int, optional): The number of financial periods to include in the analysis (default is 120)

    Attributes:
    ticker (str): The stock symbol representing the company
    period (str): Financial period used for the analysis
    metrics (dict): Dictionary of financial metrics for the company
    trends (list of plot objects): List of plots showing the trend of the financial metrics over time

    """
    def __init__(self, ticker, api_key, data='online', period='annual', limit=20, verbose=False):
        self.ticker = ticker
        self.period = period
        self.limit = limit
        self.verbose = verbose
        self._financial_data = FinancialData(ticker, api_key, data, period, limit)
        self.filing_dates = self._financial_data.filing_date_objects
        self._analysis = ManualAnalysis(self._financial_data, self.verbose)
        self.metrics = self._analysis.statement_metrics
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
        

    def recommendation(self):
        '''Built to look at the main metrics that indicate value to shareholders. Debt is also included but is handled differently
            than the equity metrics in the self.score() function
            Why these metrics: The equity ones are the main metrics regarding value to shareholders
            Debt is also important, as a company that is paying down debt is putting itself in a good position
            Tradeoff tough, as if there is no debt then there will be no decrease in debt, which will score badly.'''
        key_metrics = ['eps_growth', 'returnOnEquity_growth', 'ROIC_growth', 'returnOnAssets_growth', 
                       'debtToTotalCap_growth', 'totalDebtRatio_growth']
        scores = dict()
        for metric in key_metrics:
            score, strength = self.score(metric)
            scores[metric] = {'score': score,
                              'strength': strength}
        return scores
        

    def score(self, metric):
        if 'debt' in metric.lower():
            modifier = -1
        else:
            modifier = 1
        
        metrics = self.metrics[metric].copy()
        metrics = metrics.dropna()
        mean = metrics.mean()
        try:
            slope, intercept, r_val, _, _= linregress(range(len(metrics)), metrics)
            r_val = r_val**2
        except ValueError:
            r_val = 0

        growth_score = self.score_mean_growth(modifier * mean)
        stability_score = self.score_trend_strength(r_val)

        return (growth_score, stability_score)


    def score_mean_growth(self, mean_growth):
        growth = float(mean_growth)
        if growth <= 0.05:
            indicator = 0
        elif growth <= 0.10:
            indicator = 1
        elif growth <= 0.15:
            indicator = 2
        elif growth <= 0.2:
            indicator = 3
        else:
            indicator = 4

        return indicator
    
    def score_trend_strength(self, R2):
        R2_ = float(R2)
        if R2_ <= 0.2:
            strength = 0
        elif R2_ <= 0.3:
            strength = 1
        elif R2_ <= 0.5:
            strength = 2
        elif R2_  <= 0.75:
            strength = 3
        else:
            strength = 4
        
        return strength

    def eval_(self, scores):
        vote = 0
        for key in scores.keys():
            vote += scores[key]['score']
        # Require an average score of 1.5 for each metric
        return False if vote < 1.5*len(scores) else True 

    def print_charts(self):
        if self._charts_printed:
            return
        self._plots = Plots(self.ticker, self.period, self.metrics, self.limit, self.filing_dates)
        self.trends = self._plots.plots
        self._charts_printed = True
    
    def export(self):
        """
        Exports the financial trend charts to disk as a pdf file.
        
        """
        self._plots._export_charts_pdf()
        # also export key findings based on the analyis




        

        

        
