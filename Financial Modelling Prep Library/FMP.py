#!/usr/bin/env python
# coding: utf-8

# TODO
# - Add yfinance support to get stock price data
# - P/E ratio, dividend yield ratio, fixed charge ratio
# - Check why the ebitda calculations in Company.cross_check() are so wrong
# - create plotting functionality
# - add EPS check to cross_check() as per the calculated value from outstanding shares
# - change inventory calculations to take averages
# - total asset turnover should take average asset values

import yfinance as yf
from pandas_datareader import data as pdr
import requests
import pandas as pd
from pathlib import Path
import os

yf.pdr_override()


class Company:
    def __init__(self, ticker, api_key, data='online', period='annual', limit=120):
        data = data.lower()
        assert data in ['online', 'local'], "data must be 'online' or 'local'"
        period = period.lower()
        self.period = period
        assert period in ['annual', 'quarter'], "period must be 'annual' or 'quarter'"
        self._ticker = ticker.upper().strip()
        self.api_key = api_key

        if data == 'online':
            self.balance_sheets, self.income_statements, self.cash_flow_statements = \
                self.fetch_financial_statements(ticker, period=period, limit=limit)
            self.balance_sheets = self.build_dataframe(self.balance_sheets)
            self.income_statements = self.build_dataframe(self.income_statements)
            self.cash_flow_statements = self.build_dataframe(self.cash_flow_statements)
            assert self.balance_sheets['date'].equals(self.income_statements['date']), 'Date mismatch in financial statements'

            start_date = self.balance_sheets['date'].iloc[0]
            end_date = self.balance_sheets['date'].iloc[-1]
            price_interval = '3mo'
            self.stock_price_data = pdr.get_data_yahoo(ticker, start_date, end_date, interval=price_interval)
            

            save_path = Path.cwd()/'Company Financial Data'/ticker/period
            try:
                os.makedirs(save_path)
            except FileExistsError:
                print(f"{save_path} already exists.")
                pass
            except Exception:
                print(f'Could not create directory {save_path}')
            
            self.balance_sheets.to_parquet(save_path/'balance_sheets.parquet')
            self.balance_sheets.to_excel(save_path/'balance_sheets.xlsx')
            self.income_statements.to_parquet(save_path/'income_statements.parquet')
            self.income_statements.to_parquet(save_path/'income_statements.xlsx')
            self.cash_flow_statements.to_parquet(save_path/'cash_flow_statements.parquet')
            self.cash_flow_statements.to_parquet(save_path/'cash_flow_statements.xlsx')
            self.stock_price_data
        elif data == 'local':
            self.load_financial_statements(ticker, period)
            

        self.calculated_ratios = pd.DataFrame()
        self.metric_errors, self.ratio_errors = self.cross_check()
        self.standard_metrics = self.analyse()


    
    def fetch_financial_statements(self, company, period, limit):
        '''
        Fetches financial statements for a given company from the Financial Modeling Prep API.

        Parameters:
            - company (str): The name of the company for which financial statements are to be fetched.
            - period (str, optional): The period for which financial statements are to be fetched.
                                       Default is 'annual'.
            - limit (int, optional): The number of financial statements to be fetched.
                                      Default is 120.

        Returns:
            A tuple containing the following:
                - balance_sheets (dict): A dictionary containing the json response of the balance sheet statement API call
                - income_statements (dict): A dictionary containing the json response of the income statement API call
                - cash_flow_statements (dict): A dictionary containing the json response of the cash flow statement API call
        '''

        balance_sheets = requests.get(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?period={period}&limit={limit}&apikey={self.api_key}')
        income_statements = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{company}?period={period}&limit={limit}&apikey={self.api_key}')
        cash_flow_statements = requests.get(f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{company}?period={period}&limit={limit}&apikey={self.api_key}')

        return balance_sheets.json(), income_statements.json(), cash_flow_statements.json()


    

    def load_financial_statements(self, ticker, period):
        '''
        Load from a local directory
        draw distinction between annual and quartely results using an f-string'''
        load_path = Path.cwd()/'Company Financial Data'/ticker/period
        self.income_statements = pd.read_parquet(load_path/'income_statements.parquet')
        self.balance_sheets = pd.read_parquet(load_path/'balance_sheets.parquet')
        self.cash_flow_statements = pd.read_parquet(load_path/'cash_flow_statements.parquet')


    
    
    def build_dataframe(self, statements):
        '''
        Builds a pandas DataFrame from a list of financial statements.
        Asserts that all statements have the same keys.

        Parameters:
            - statements (List[Dict[str, Any]]): A list of financial statements, where each statement is a dictionary 
                                                 with keys and values representing the financial data.

        Returns:
            A pandas DataFrame containing the financial statements, with columns corresponding to the keys of the dictionaries,
            and rows corresponding to the values of the dictionaries.
        '''

        keys = set(statements[0].keys())
        for statement in statements:
            assert set(statement.keys()) == keys
        data = []
        for statement in reversed(statements):
            data.append(list(statement.values()))
        df = pd.DataFrame(data, columns = statements[0].keys())
        df['index'] = df['date'].apply(lambda x: self.generate_index(x))
        df.set_index('index', inplace=True)
        if 'netIncome' and 'eps' in df.keys():
            df['outstandingShares_calc'] = df['netIncome']/df['eps']
        return df


    def cross_check(self):
        '''Returns metric_errors, ratio_errors as dataframes'''
        self._check_reported_values = pd.DataFrame()
        self._check_calculated_values = pd.DataFrame()
        RND_expenses = self.income_statements['researchAndDevelopmentExpenses']
        SGA_expenses = self.income_statements['sellingGeneralAndAdministrativeExpenses']
        other_expenses = self.income_statements['otherExpenses']
        revenue = self.income_statements['revenue']
        self.revenue = revenue
        cost_of_revenue = self.income_statements['costOfRevenue']
        depreciation_amortization = self.cash_flow_statements['depreciationAndAmortization']
        interest_expense = self.income_statements['interestExpense']
        interest_income = self.income_statements['interestIncome']


        metrics = ['ebitda', 'ebitdaratio', 'grossProfit', 'grossProfitRatio', 'operatingIncome', 'operatingIncomeRatio', \
                    'incomeBeforeTax', 'incomeBeforeTaxRatio', 'netIncome', 'netIncomeRatio']
        ratios = [i if 'ratio' in i.lower() else None for i in metrics]

        # Calculated ratios from reported values on the financial statements
        self._check_calculated_values['ebitda'] = revenue - RND_expenses - SGA_expenses - other_expenses + depreciation_amortization
        self._check_calculated_values['ebitdaratio'] = self._check_calculated_values['ebitda']/revenue
        self._check_calculated_values['grossProfit'] = revenue - cost_of_revenue
        self._check_calculated_values['grossProfitRatio'] = (self._check_calculated_values['grossProfit']/revenue)
        self._check_calculated_values['operatingIncome'] = revenue - cost_of_revenue - SGA_expenses - RND_expenses
        self._check_calculated_values['operatingIncomeRatio'] = (self._check_calculated_values['operatingIncome']/revenue)
        self._check_calculated_values['incomeBeforeTax'] = self._check_calculated_values['operatingIncome'] - interest_expense + interest_income
        self._check_calculated_values['incomeBeforeTaxRatio'] = (self._check_calculated_values['incomeBeforeTax']/revenue)
        self._check_calculated_values['netIncome'] = 0.79*self._check_calculated_values['incomeBeforeTax']
        self._check_calculated_values['netIncomeRatio'] = (self._check_calculated_values['netIncome']/revenue)
        
        # Pulling reported metric values
        for metric in metrics:
            if metric in self.income_statements.keys():
                self._check_reported_values[metric] = self.income_statements[metric]
            elif metric in self.balance_sheets.keys():
                self._check_reported_values[metric] = self.balance_sheets[metric]
            elif metric in self.cash_flow_statements.keys():
                self._check_reported_values[metric] = self.cash_flow_statements[metric]

        # Ensuring dimensionality of the two dataframes
        if len(self._check_calculated_values.keys()) != len(self._check_reported_values.keys()):
            msg = '''Key mismatch between the reported and calculated tables.\nCheck the calculations in the Company.cross_check() method'''
            raise Exception(msg)
        metric_errors = self._check_calculated_values - self._check_reported_values
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
        '''Stock Evaluation Ratios'''
        self.calculated_ratios['eps_calc'] = self.income_statements['netIncome']/self.balance_sheets['commonStock'] #authorized stock!!!
        self.calculated_ratios['eps_reported'] = self.income_statements['eps']
        self.calculated_ratios['eps_diluted'] = self.income_statements['epsdiluted']
        ### Add P/E ratio here
        self.calculated_ratios['bookValuePerShare'] = (self.balance_sheets['totalAssets']-self.balance_sheets['totalLiabilities'])/self.income_statements['outstandingShares_calc']
        self.calculated_ratios['dividendPayoutRatio'] = self.cash_flow_statements['dividendsPaid']/self.income_statements['eps']
        ### Add dividend yield 
        
        '''Profitability Ratios'''
        self.calculated_ratios['grossProfitMargin'] = self.income_statements['grossProfit']/self.income_statements['revenue']
        self.calculated_ratios['operatingIncome_calc'] = self.income_statements['revenue'] \
                                                            - self.income_statements['costOfRevenue'] \
                                                            - self.income_statements['sellingGeneralAndAdministrativeExpenses'] \
                                                            - self.income_statements['researchAndDevelopmentExpenses']
        self.calculated_ratios['operatingProfitMargin'] = self.income_statements['operatingIncome']/self.income_statements['revenue']
        self.calculated_ratios['pretaxProfitMargin'] = self.income_statements['incomeBeforeTax']/self.income_statements['revenue']
        
        total_capitalization = self.balance_sheets['totalEquity'] + self.balance_sheets['longTermDebt']
        net_income = self.income_statements['netIncome']
        self.calculated_ratios['ROIC'] = net_income/total_capitalization
        self.calculated_ratios['ROE'] = net_income/self.balance_sheets['totalStockholdersEquity']
        self.calculated_ratios['ROA'] = net_income/self.balance_sheets['totalAssets']

        '''Debt and Interest Ratios'''
        self.calculated_ratios['interestCoverage'] = self.income_statements['operatingIncome']/self.income_statements['interestExpense']
        
        # FIX (haha) the fixed charge ratio below
        fixed_charges = self.income_statements['interestExpense'] # +  XXX
        # self.calculated_ratios['fixed_charge_coverage'] = self.income_statements['ebitda']/

        self.calculated_ratios['debtToTotalCap'] = self.balance_sheets['longTermDebt']/total_capitalization

        total_debt = self.balance_sheets['totalAssets'] - self.balance_sheets['totalEquity']
        self.calculated_ratios['totalDebtRatio'] = total_debt/self.balance_sheets['totalAssets']

        '''Liquidity & FinancialCondition Ratios'''
        self.calculated_ratios['currentRatio'] = self.balance_sheets['totalCurrentAssets']/self.balance_sheets['totalCurrentLiabilities']
        quick_assets = self.balance_sheets['totalCurrentAssets'] - self.balance_sheets['inventory']
        self.calculated_ratios['quickRatio'] = quick_assets/self.balance_sheets['totalCurrentLiabilities']
        self.calculated_ratios['cashRatio'] = self.balance_sheets['cashAndCashEquivalents']/self.balance_sheets['totalCurrentLiabilities']


        '''Efficiency Ratios'''
        self.calculated_ratios['totalAssetTurnover'] = self.income_statements['revenue']/self.balance_sheets['totalAssets']
        self.calculated_ratios['inventoryToSalesRatio'] = self.balance_sheets['inventory']/self.income_statements['revenue']
        self.calculated_ratios['inventoryTurnoverRatio'] = 1/self.calculated_ratios['inventoryToSalesRatio']
        days = 365 if self.period == 'annual' else 90
        self.calculated_ratios['inventoryTurnoverInDays'] = days/self.calculated_ratios['inventoryTurnoverRatio']

        accounts_receivable_to_sales_ratio = self.balance_sheets['netReceivables']/self.income_statements['revenue']
        self.calculated_ratios['accountsReceivableToSalesRatio'] = accounts_receivable_to_sales_ratio
        self.calculated_ratios['receivablesTurnover'] = self.income_statements['revenue']/self.balance_sheets['netReceivables']
        self.calculated_ratios['receivablesTurnoverInDays'] = days/self.calculated_ratios['receivablesTurnover']







        
        

    def buffet_analysis():
        '''perform analysis as per Buffetology'''
        pass


    def generate_index(self, date):
        '''
        Generates a financial index for a company based on a given date.

        Parameters:
            - date (str): A date in the format 'yyyy-mm-dd'.

        Returns:
            A string in the format '{ticker}-Q{quarter}-{year}' or '{ticker}-FY-{year}, where ticker is the ticker of the company, quarter is the quarter 
            of the date (1-4) and year is the year of the date.
        '''
        year, month, _ = [int(i) for i in date.split('-')]
        
        if self.period == 'annual':
            return f"{self._ticker}-FY-{year}"
            
        if month in (1,2,3):
            quarter = 1
        elif month in (4,5,6):
            quarter = 2
        elif month in (7,8,9):
            quarter = 3
        elif month in (10, 11, 12):
            quarter = 4
        
        return f"{self._ticker}-Q{quarter}-{year}"
        


