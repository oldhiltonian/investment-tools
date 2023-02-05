import yfinance as yf
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import requests
import pandas as pd
from pathlib import Path
import os

yf.pdr_override()

class FinancialData:
    def __init__(self, ticker, api_key='', data='local', period='annual', limit=120):
        self.ticker = ticker.upper().strip()
        self.api_key = str(api_key)
        self.data = data.lower().strip()
        self.period = period.lower().strip()
        self.limit = int(limit)
        self.days_in_period = 356 if period == 'annual' else 90


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
            else:
                self._frame_indecies = self.balance_sheets.index
            
            self.stock_price_data = self.fetch_stock_price_data()
            self.save_financial_attributes()

        elif data == 'local':
            self.load_financial_statements(ticker, period)

        

    def fetch_financial_statements(self, company, api_key, period, limit):
        # need to throw an exception here if the API returns an error
        # test that a tuple of json objects is returned
        balance_sheets = requests.get(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?period={period}&limit={limit}&apikey={api_key}')
        income_statements = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{company}?period={period}&limit={limit}&apikey={api_key}')
        cash_flow_statements = requests.get(f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{company}?period={period}&limit={limit}&apikey={api_key}')
        return balance_sheets.json(), income_statements.json(), cash_flow_statements.json()


    def build_dataframe(self, statements):
        # throw an exception if statemetns != List(dict)
        keys = set(statements[0].keys())
        for statement in statements:
            assert set(statement.keys()) == keys, 'column mismatch across financial statement'
        data = []
        for statement in reversed(statements):
            data.append(list(statement.values()))
        df = pd.DataFrame(data, columns = statements[0].keys())
        df['index'] = df['date'].apply(lambda x: self.generate_index(x))
        # Don;t like saving the date here as it happens 3 times
        self.filing_date_strings = df['date']
        df['date'] = df['date'].apply(lambda x: self.generate_date(x))
        self.filing_date_objects = df['date']
        df = df.set_index('index')
        # this should be stripped out to a method just called on the income statement
        if 'netIncome' and 'eps' in df.keys():
            df['outstandingShares_calc'] = df['netIncome']/df['eps']
        return df


    def generate_index(self, date):
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
        year, month, day = [int(i) for i in date_str.split()[0].split('-')]
        return dt.date(year, month, day)


    def filter_for_common_indecies(self):
        idx1 = self.balance_sheets.index
        idx2 = self.income_statements.index
        idx3 = self.cash_flow_statements.index
        common_elements = idx1.intersection(idx2).intersection(idx3)
        self.balance_sheets = self.balance_sheets.loc[common_elements]
        self.income_statements = self.income_statements.loc[common_elements]
        self.cash_flow_statements = self.cash_flow_statements.loc[common_elements]


    def fetch_stock_price_data(self):
            '''Need to catch if the request fails or returns a null frame'''
            start_date = dt.date(*[int(i) for i in self.filing_date_strings.iloc[0].split('-')])
            end_date = dt.date(*[int(i) for i in self.filing_date_strings.iloc[-1].split('-')])
            price_interval = '1d'
            raw_data = pdr.get_data_yahoo(self.ticker, start_date, end_date, interval=price_interval)
            raw_data['date'] = raw_data.index.date
            return self.periodise(raw_data)

    def periodise(self, df):
        working_array = []
        for i in range(len(self.filing_date_strings)):
            if i == 0 or i == len(self.filing_date_objects):
                working_array.append([np.nan]*3)
                continue

            period_data = df[(df['date'] >= self.filing_date_objects.iloc[i-1]) & (df['date'] < self.filing_date_objects.iloc[i])]
            # print(period_data)
            max_price = max(period_data['High'])
            min_price = min(period_data['Low'])
            avg_close = period_data['Close'].mean()
            working_array.append([max_price, min_price, avg_close])

        cols = ['high', 'low', 'avg_close']
        periodised = pd.DataFrame(working_array, index=self._frame_indecies, columns=cols)
        assert sum(periodised['high'] < periodised['low']) <= 1, 'Stock highs and lows not consistent'
        return periodised


    def save_financial_attributes(self):
        save_path = Path.cwd()/'Company Financial Data'/self.ticker/self.period
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print(f"{save_path} already exists. Overwriting data.")
        except Exception:
            msg = 'Could not create directory {save_path}'
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
        '''
        Load from a local directory
        draw distinction between annual and quartely results using an f-string'''
        load_path = Path.cwd()/'Company Financial Data'/self.ticker/self.period
        self.income_statements = pd.read_parquet(load_path/'income_statements.parquet')
        self.balance_sheets = pd.read_parquet(load_path/'balance_sheets.parquet')
        self.cash_flow_statements = pd.read_parquet(load_path/'cash_flow_statements.parquet')
        self.stock_price_data = pd.read_parquet(load_path/'stock_price_data.parquet')

class Analysis:
    def __init__(self, financial_data):
        self.data = financial_data
        self.metric_errors, self.ratio_errors = self.cross_check()
        self.metrics = self.analyse()



    def cross_check(self):
        '''Returns metric_errors, ratio_errors as dataframes'''
        reported_metrics = pd.DataFrame()
        calculated_metrics = pd.DataFrame()
        RND_expenses = self.data.income_statements['researchAndDevelopmentExpenses']
        SGA_expenses = self.data.income_statements['sellingGeneralAndAdministrativeExpenses']
        other_expenses = self.data.income_statements['otherExpenses']
        revenue = self.data.income_statements['revenue']
        self.revenue = revenue
        cost_of_revenue = self.data.income_statements['costOfRevenue']
        depreciation_amortization = self.data.cash_flow_statements['depreciationAndAmortization']
        interest_expense = self.data.income_statements['interestExpense']
        interest_income = self.data.income_statements['interestIncome']


        metrics = ['ebitda', 'ebitdaratio', 'grossProfit', 'grossProfitRatio', 'operatingIncome', 'operatingIncomeRatio', \
                    'incomeBeforeTax', 'incomeBeforeTaxRatio', 'netIncome', 'netIncomeRatio', 'eps']
        ratios = [i if 'ratio' in i.lower() else None for i in metrics]

        # Calculated ratios from reported values on the financial statements
        calculated_metrics['ebitda'] = revenue - cost_of_revenue- RND_expenses - SGA_expenses - other_expenses + depreciation_amortization
        calculated_metrics['ebitdaratio'] = calculated_metrics['ebitda']/revenue
        calculated_metrics['grossProfit'] = revenue - cost_of_revenue
        calculated_metrics['grossProfitRatio'] = (calculated_metrics['grossProfit']/revenue)
        calculated_metrics['operatingIncome'] = revenue - cost_of_revenue - SGA_expenses - RND_expenses
        calculated_metrics['operatingIncomeRatio'] = (calculated_metrics['operatingIncome']/revenue)
        calculated_metrics['incomeBeforeTax'] = calculated_metrics['operatingIncome'] - interest_expense + interest_income
        calculated_metrics['incomeBeforeTaxRatio'] = (calculated_metrics['incomeBeforeTax']/revenue)
        calculated_metrics['netIncome'] = 0.79*calculated_metrics['incomeBeforeTax']
        calculated_metrics['netIncomeRatio'] = (calculated_metrics['netIncome']/revenue)
        calculated_metrics['eps'] = calculated_metrics['netIncome']/self.data.income_statements['outstandingShares_calc']

        
        # Pulling reported metric values
        for metric in metrics:
            if metric in self.data.income_statements.keys():
                reported_metrics[metric] = self.data.income_statements[metric]
            elif metric in self.balance_sheets.keys():
                reported_metrics[metric] = self.data.balance_sheets[metric]
            elif metric in self.data.cash_flow_statements.keys():
                reported_metrics[metric] = self.data.cash_flow_statements[metric]

        # Ensuring dimensionality of the two dataframes
        if len(calculated_metrics.keys()) != len(reported_metrics.keys()):
            msg = '''Key mismatch between the reported and calculated tables.\nCheck the calculations in the Company.cross_check() method'''
            raise Exception(msg)
        metric_errors = calculated_metrics - reported_metrics
        ratio_errors = metric_errors.drop(['ebitda','grossProfit', 'operatingIncome', 'incomeBeforeTax', 'netIncome'], inplace=False, axis=1)
        # Error between calculated and reported values
        error_tolerance = 0.05
        error_messages  = []
        line_count = len(ratio_errors)
        for ratio in ratios:
            if ratio is None:
                continue
            count = sum(ratio_errors[ratio] >= error_tolerance)
            error_messages.append(f"There were {count}/{line_count} values in {ratio} that exceed the {error_tolerance} error tolerance.")
        for message in error_messages:
            print(message)

        return metric_errors, ratio_errors

    def analyse(self):
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
        df['dividend_yield_low'] = (-dividends_paid/outstanding_shares)/stock_price_high
        df['dividend_yield_high'] = (-dividends_paid/outstanding_shares)/stock_price_low 
        df['dividend_yield_avg_close'] = (-dividends_paid/outstanding_shares)/stock_price_avg
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
        df['operatingIncome_calc'] = revenue - COGS - SGA - RND_expense
        df['operatingProfitMargin'] = operating_income/revenue
        df['pretaxProfitMargin'] = income_before_tax/revenue
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
        df['fixed_charge_coverage'] = ebitda/fixed_charges
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


class Company:
    def __init__(self, ticker, api_key, data='online', period='annual', limit=120):
        self.financial_data = FinancialData(ticker, api_key, data, period, limit)
        self.analysis = Analysis(self.financial_data)
        self.metrics = self.analysis.metrics
