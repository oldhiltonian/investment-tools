### TODO
### - Fix plot x-axis to be a more inteligible value: FY22, or 2Q22
### - Add R2 values to each appropriate subplot for correlation strengths and trendlines
### - Add functionality for the user to select how far back to look
### - Limit appropriate ratios to between -1 and 1


import yfinance as yf
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import requests
import pandas as pd
from pathlib import Path
import os

yf.pdr_override()

class FinancialData:
    """
    A class for handling financial data

    The FinancialData class is used for fetching and processing financial data for
    a given company. The data can be fetched from either a local source or from the 
    Financial Modeling Prep API (https://financialmodelingprep.com/developer/docs/). 
    
    Args:
    - ticker (str): Ticker symbol for the company to fetch financial data for
    - api_key (str, optional): API key for accessing the Financial Modeling Prep API. 
        Default is an empty string.
    - data (str, optional): Data source. Valid options are 'local' and 'online'. 
        Default is 'local'.
    - period (str, optional): Period of the financial data to retrieve. Valid options 
        are 'annual' and 'quarterly'. Default is 'annual'.
    - limit (int, optional): Maximum number of financial records to retrieve. 
        Default is 120.

    Attributes:
    - ticker (str): Ticker symbol for the company to fetch financial data for
    - api_key (str): API key for accessing the Financial Modeling Prep API.
    - data (str): Data source. Can be 'local' or 'online'.
    - period (str): Period of the financial data to retrieve. Can be 'annual' or 'quarterly'.
    - limit (int): Maximum number of financial records to retrieve.
    - days_in_period (int): The number of days in a period (90 for quarterly, 356 for annual)
    - balance_sheets (pandas.DataFrame): Balance sheets for the company
    - income_statements (pandas.DataFrame): Income statements for the company
    - cash_flow_statements (pandas.DataFrame): Cash flow statements for the company
    - filing_date_objects (pandas.Series): A series of date objects, one for each 
        inancial statement.
    - stock_price_data (pandas.DataFrame): Stock price data for the company
    """
    def __init__(self, ticker, api_key='', data='local', period='annual', limit=120):
        self.ticker = ticker.upper().strip()
        self.api_key = str(api_key)
        self.data = data.lower().strip()
        self.period = period.lower().strip()
        self.limit = int(limit)
        self.days_in_period = 365 if period == 'annual' else 90


        if data == 'online':
            bs, is_, cfs = self.fetch_financial_statements(ticker, api_key, period, limit)
            self.balance_sheets = self.build_dataframe(bs)
            self.income_statements = self.build_dataframe(is_)
            self.cash_flow_statements = self.build_dataframe(cfs)
            self.days_in_period = 356 if period == 'annual' else '90'
            
            matching_index_1 =  self.balance_sheets['date'].equals(self.income_statements['date'])
            matching_index_2 = self.balance_sheets['date'].equals(self.cash_flow_statements['date'])
            matching_indecies = matching_index_1 and matching_index_2
            if not matching_indecies:
                self.filter_for_common_indecies()
            self._frame_indecies = self.balance_sheets.index
            self.filing_date_objects = self.balance_sheets['date']
            self.stock_price_data = self.fetch_stock_price_data()
            self.save_financial_attributes()

        elif data == 'local':
            self.load_financial_statements(ticker, period)

        

    def fetch_financial_statements(self, company, api_key, period, limit):
        # need to throw an exception here if the API returns an error
        # test that a tuple of json objects is returned
        """
        This function fetches the balance sheet, income statement, and cash flow statement
        for a given company, using the provided api_key, period, and limit.
        
        Args:
            company (str): The company's ticker symbol or name.
            api_key (str): The API key to be used to access financial data.
            period (str): The reporting period to retrieve data for, e.g. 'quarter' or 'annual'.
            limit (int): The number of reporting periods to retrieve.
            
        Returns:
            Tuple[Dict, Dict, Dict]: The balance sheet, income statement, and cash flow statement
            as a tuple of dictionaries in JSON format.
            
        Raises:
            Exception: If the API returns an error.
            
        Example:
            statements = fetch_financial_statements('AAPL', 'apikey', 'quarter', 4)
            print(statements)
        """

        balance_sheets = requests.get(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?period={period}&limit={limit}&apikey={api_key}')
        assert balance_sheets.status_code == 200, f"API call failed. Code <{balance_sheets.status_code}>"
        income_statements = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{company}?period={period}&limit={limit}&apikey={api_key}')
        assert income_statements.status_code == 200, f"API call failed. Code <{income_statements.status_code}>"
        cash_flow_statements = requests.get(f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{company}?period={period}&limit={limit}&apikey={api_key}')
        assert cash_flow_statements.status_code == 200, f"API call failed. Code <{cash_flow_statements.status_code}>"

        return balance_sheets.json(), income_statements.json(), cash_flow_statements.json()


    def build_dataframe(self, statements):
        # throw an exception if statemetns != List(dict)
        """
        This function builds a pandas dataframe from a list of financial statement dictionaries.
        
        Args:
            statements (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents
            a financial statement.
            
        Returns:
            pandas.DataFrame: A dataframe with the financial statement data, indexed by a generated date.
            
        Raises:
            AssertionError: If the statements passed in as argument is not a list of dictionaries or if
            there is a mismatch in the columns across the financial statements.
            
        Example:
            statements = [{'date': '2022-01-01', 'revenue': 100, 'costs': 80},
                        {'date': '2022-02-01', 'revenue': 120, 'costs': 90}]
            df = build_dataframe(statements)
        """
        err_msg = "Empty statement. Perhaps check the .json() conversion off of the API response"
        assert len(statements) > 0, err_msg

        keys = set(statements[0].keys())
        for statement in statements:
            assert set(statement.keys()) == keys, 'column mismatch across financial statement'
        data = []
        for statement in reversed(statements):
            data.append(list(statement.values()))
        df = pd.DataFrame(data, columns = statements[0].keys())
        df['index'] = df['date'].apply(lambda x: self.generate_index(x))
        df['date'] = df['date'].apply(lambda x: self.generate_date(x))
        df = df.set_index('index')
        if 'netIncome' and 'eps' in df.keys():
            df['outstandingShares_calc'] = df['netIncome']/df['eps']
        return df


    def generate_index(self, date):
        """
        Generate an index string from the date string
        
        Parameters:
        -----------
        date : str
            Date string in format "YYYY-MM-DD".
        
        Returns:
        -------
        str
            Index string in the format "ticker-FY/QX-YYYY".
        
        Example:
        -------
        generate_index("2021-06-15") returns "ticker-Q2-2021"
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
        Generate a date object from the date string
        
        Parameters:
        -----------
        date_str : str
            Date string in format "YYYY-MM-DD".
        
        Returns:
        -------
        date
            Date object.
        
        Example:
        -------
        generate_date("2021-06-15") returns date object "2021-06-15"
        """
        year, month, day = [int(i) for i in date_str.split()[0].split('-')]
        return dt.date(year, month, day)


    def filter_for_common_indecies(self):
        """
        Filter for common indices between balance sheets, 
            income statements, and cash flow statements.

        This method is used to ensure that all financial statement dataframes have the
            same set of indices. If the length of the dataframes is different, the 
            method will drop any missing indices from the dataframes to ensure consistency.

        Returns:
            None
        """
        print('Financial statements had different lengths...')
        print(f"Financial statement lengths are BS: {len(self.balance_sheets)}, IS:{len(self.income_statements)}, CFS:{len(self.cash_flow_statements)}")
        idx1 = self.balance_sheets.index
        idx2 = self.income_statements.index
        idx3 = self.cash_flow_statements.index
        common_elements = idx1.intersection(idx2).intersection(idx3)
        self.balance_sheets = self.balance_sheets.loc[common_elements]
        self.income_statements = self.income_statements.loc[common_elements]
        self.cash_flow_statements = self.cash_flow_statements.loc[common_elements]
        assert len(self.cash_flow_statements) == len(self.balance_sheets), 'Indecies could not be filtered for common elements'
        assert len(self.income_statements) == len(self.balance_sheets), 'Indecies could not be filtered for common elements'
        print(f"Financial statement lengths are now each: {len(self.balance_sheets)}")



    def fetch_stock_price_data(self):

        # '''Need to catch if the request fails or returns a null frame'''
        """
        This function fetches stock price data for the company using the provided data source.

        Returns:
        - pandas.DataFrame: Stock price data for the company in the form of a pandas dataframe.
        
        """
        start_date = dt.date(*[int(i) for i in str(self.filing_date_objects.iloc[0]).split()[0].split('-')]) - dt.timedelta(days=370)
        # end_date = dt.date(*[int(i) for i in str(self.filing_date_objects.iloc[-1]).split()[0].split('-')])
        end_date = dt.date.today()
        price_interval = '1d'
        raw_data = pdr.get_data_yahoo(self.ticker, start_date, end_date, interval=price_interval)
        raw_data['date'] = raw_data.index.date
        return self.periodise(raw_data)

    def periodise(self, df):
        """
        This function takes a time series and a period as input and returns a new time series with the data aggregated to the given period.

        Parameters:
        timeseries (list or numpy array): The input time series data, represented as a list or a numpy array of numerical values.
        period (int): The desired period of the aggregated time series. This should be a positive integer representing the number of time units in the desired period.

        Returns:
        numpy array: A pandas DataFrame containing the aggregated time series.

        Example:
        >>> periodise([1, 2, 3, 4, 5, 6], 2)
        array([1.5, 3.5, 5.5])
        >>> periodise([1, 2, 3, 4, 5, 6, 7, 8], 3)
        array([2., 5., 8.])
        """
        working_array = []
        for i in range(len(self.filing_date_objects)):
            if i == 0 or i == len(self.filing_date_objects):
                working_array.append([np.nan]*3)
                continue
            period_data = df[(df['date'] >= self.filing_date_objects.iloc[i-1]) & (df['date'] < self.filing_date_objects.iloc[i])]
            try:
                max_price = max(period_data['High'])
                min_price = min(period_data['Low'])
                avg_close = period_data['Close'].mean()
            except ValueError:
                working_array.append([np.nan]*3)
            else:
                working_array.append([max_price, min_price, avg_close])

        cols = ['high', 'low', 'avg_close']
        periodised = pd.DataFrame(working_array, index=self._frame_indecies, columns=cols)
        assert sum(periodised['high'] < periodised['low']) <= 1, 'Stock highs and lows not consistent'
        return periodised


    def save_financial_attributes(self):
        """
        This function saves the financial attributes of the object to disk as a pickle file.        
        """
        save_path = Path.cwd()/'Company Financial Data'/self.ticker/self.period
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print(f"Path already exists. Overwriting saved data.")
        except Exception:
            msg = f'Could not create directory {save_path}'
            raise Exception(msg)
        
        self.balance_sheets.to_parquet(save_path/'balance_sheets.parquet')
        self.balance_sheets.to_excel(save_path/'balance_sheets.xlsx')
        self.income_statements.to_parquet(save_path/'income_statements.parquet')
        self.income_statements.to_excel(save_path/'income_statements.xlsx')
        self.cash_flow_statements.to_parquet(save_path/'cash_flow_statements.parquet')
        self.cash_flow_statements.to_excel(save_path/'cash_flow_statements.xlsx')
        self.stock_price_data.to_parquet(save_path/'stock_price_data.parquet')
        self.stock_price_data.to_excel(save_path/'stock_price_data.xlsx')


    def load_financial_statements(self):
        """
        This function loads financial statements from disk.
        
        Args:
        - company (str): The company's ticker symbol or name.
        - period (str): The reporting period of the financial statements to load, e.g. 'quarter' or 'annual'.
        
        """
        load_path = Path.cwd()/'Company Financial Data'/self.ticker/self.period
        self.income_statements = pd.read_parquet(load_path/'income_statements.parquet')
        self.balance_sheets = pd.read_parquet(load_path/'balance_sheets.parquet')
        self.cash_flow_statements = pd.read_parquet(load_path/'cash_flow_statements.parquet')
        self.stock_price_data = pd.read_parquet(load_path/'stock_price_data.parquet')



class Analysis:
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
    def __init__(self, financial_data):
        self.data = financial_data
        clc, rep, met_err, rat_err = self.cross_check()
        self.calculated_metrics = clc
        self.reported_metrics= rep
        self.metric_errors = met_err
        self.ratio_errors = rat_err
        self.metrics = self.analyse()



    def cross_check(self):
        """
        Calculates financial metrics and and compares them to the reported values.
        
        Returns pandas DataFrames representing the calculated values, reported values,
        the error between the calculated and reported values, and the error of just
        the financial ratios"""
    
        reported = pd.DataFrame()
        calculated = pd.DataFrame()
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
        ratios = [i if 'ratio' in i.lower() else None for i in metrics] + ['eps']

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
        
        error_tolerance = 0.05
        error_messages  = []
        line_count = len(ratio_errors)
        for ratio in ratios:
            if ratio is not None:
                count = sum(ratio_errors[ratio] >= error_tolerance)
                error_messages.append(f"There were {count}/{line_count} values in {ratio} that exceed the {error_tolerance} error tolerance.")
        for message in error_messages:
            print(message)

        return calculated, reported, metric_errors, ratio_errors

    def analyse(self):
        """Calculates and returns important financial metrics and ratios as a 
            pandas DataFrame."""
        df = pd.DataFrame()

        '''Stock Evaluation Ratios'''
        total_assets = self.data.balance_sheets['totalAssets']
        total_liabilities = self.data.balance_sheets['totalLiabilities']
        dividends_paid = self.data.cash_flow_statements['dividendsPaid']
        outstanding_shares = self.data.income_statements['outstandingShares_calc']
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
        df['editdaratio'] = self.data.income_statements['ebitdaratio']


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
        df['ROE'] = net_income/total_shareholder_equity
        df['ROA'] = net_income/total_assets

        '''Debt and Interest Ratios'''
        interest_expense = self.data.income_statements['interestExpense']
        # The fixed_charges calculation below is likely incomplete
        fixed_charges = self.data.income_statements['interestExpense'] + self.data.balance_sheets['capitalLeaseObligations']
        ebitda = self.data.income_statements['ebitda']
        long_term_debt = self.data.balance_sheets['longTermDebt']
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
        cash_and_equivalents = self.data.balance_sheets['cashAndCashEquivalents']
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
        return df


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
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.ratio_dict = {
            'Stock Evaluation Ratios': ['eps', 'eps_diluted', 'PE_high', 'PE_low', 'PE_avg_close', \
                                'bookValuePerShare', 'dividendPayoutRatio', 'dividendYield_avg_close'],

            'Profitability Ratios': ['grossProfitMargin', 'operatingProfitMargin', 'pretaxProfitMargin', 'netProfitMargin',\
                                     'ROIC', 'ROE', 'ROA'],

            'Debt & Interest Ratios': ['interestCoverage', 'fixedChargeCoverage', 'debtToTotalCap', 'totalDebtRatio'],

            'Liquidity Ratios': ['currentRatio', 'quickRatio', 'cashRatio'],

            'Efficiency Ratios': ['totalAssetTurnover', 'inventoryToSalesRatio', 'inventoryTurnoverRatio', \
                                  'inventoryTurnoverInDays', 'accountsReceivableToSalesRatio', 'receivablesTurnover', \
                                  'receivablesTurnoverInDays']
        }

        self.plots = []
        self.plot()

    def plot(self):
        """
        Plots the financial ratios using matplotlib.

        Returns:
        None
        """
        for ratio_type in self.ratio_dict.keys():
            nplots = len(self.ratio_dict[ratio_type])
            nrows = -(-nplots//2)
            fig, ax = plt.subplots(nrows, 2, figsize=(11.7, 8.3))
            for counter, ratio in enumerate(self.ratio_dict[ratio_type]):
                i, j = counter//2, counter%2
                ax[i][j].plot(self.data[ratio][-self.n:])
                ax[i][j].set_title(ratio)
            fig.suptitle(ratio_type)
            fig.tight_layout()
            self.plots.append(fig)




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
    def __init__(self, ticker, api_key, data='online', period='annual', limit=120):
        self.ticker = ticker
        self.period = period
        self._financial_data = FinancialData(ticker, api_key, data, period, limit)
        self._analysis = Analysis(self._financial_data)
        self.metrics = self._analysis.metrics
        self._plots = Plots(self.metrics, 5)
        self.trends = self._plots.plots

    def export(self):
        # why did I make this method again??
        """
        Exports the financial trend charts to disk as a pdf file.
        
        """
        self.print_charts()

    def print_charts(self):
        """
        Creates the financial trend charts as a pdf file.
        
        """
        title_message = f"Financial Ratio Trends for {self.ticker}"
        title_page, ax = plt.subplots(1,1, figsize=(11.7, 8.3))
        title_page.suptitle(title_message)
        self.figures = [title_page] + self.trends
        end_date = self._financial_data.filing_date_objects[-1]
        start_date = self._financial_data.filing_date_objects[0]
        file_name = f"{self.ticker}_{self.period}__{str(start_date)}_to_{str(end_date)}.pdf"
        file_path = Path.cwd()/'Company Analysis'/self.ticker/self.period
        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

        with PdfPages(filename=file_path/file_name) as pdf:
            for figure in self.figures:
                pdf.savefig(figure)