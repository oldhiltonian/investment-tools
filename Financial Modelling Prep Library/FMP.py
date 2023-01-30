#!/usr/bin/env python
# coding: utf-8

# TODO
# - Add yfinance support to get stock price data
# - Check why the ebitda calculations in Company.cross_check() are so wrong
# - create load_financial_statements()
# - expand build_dataframe() to automatically save to excel/
# - create analyse() to perform all calculations, or perhaps to literally do everything start-to-finish
# - create plotting functionality
# - add the following to Company.analyse(): all matrics from my personal notes from 5.2.2 Operating profit margin onwards

import requests
import pandas as pd
from pathlib import Path
import os


class Company:
    def __init__(self, ticker, api_key, data='online', period='annual', limit=120):
        self._ticker = ticker
        self.api_key = api_key
        if data == 'online':
            self.balance_sheets, self.income_statements, self.cash_flow_statements = \
                self.fetch_financial_statements(ticker, period=period, limit=limit)
            self.balance_sheets = self.build_dataframe(self.balance_sheets)
            self.income_statements = self.build_dataframe(self.income_statements)
            self.cash_flow_statements = self.build_dataframe(self.cash_flow_statements)
            save_path = Path.cwd()/'Company Financial Data'/ticker/period
            try:
                os.makedirs(save_path)
            except Exception(save_path):
                print(f"Creation of the directory {save_path} failed.")
            
            self.balance_sheets.to_parquet(save_path/'balance_sheets.parquet')
            self.balance_sheets.to_excel(save_path/'balance_sheets.xlsx')
            self.income_statements.to_parquet(save_path/'income_statements.parquet')
            self.income_statements.to_parquet(save_path/'income_statements.xlsx')
            self.cash_flow_statements.to_parquet(save_path/'cash_flow_statements.parquet')
            self.cash_flow_statements.to_parquet(save_path/'cash_flow_statements.xlsx')

        elif data == 'local':
            self.load_financial_statements(ticker, period)
        
        else:
            msg = 'Something went wrong in loading the data'
            raise Exception(msg)
            

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
        df.drop(['date', 'symbol'], axis=1, inplace=True)
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
        self._check_calculated_values['incomeBeforeTax'] = self._check_calculated_values['operatingIncome'] - interest_expense
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
        self.calculated_ratios['bookValuePerShare'] = (self.balance_sheets['totalAssets']-self.balance_sheets['totalLiabilities'])                                                       /self.income_statements['outstandingShares_calc']
        self.calculated_ratios['dividendPayoutRatio'] = self.cash_flow_statements['dividendsPaid']/self.income_statements['eps']
        ### Add dividend yield 
        
        '''Profitability Ratios'''
        self.calculated_ratios['grossProfit_calc'] = self.income_statements['revenue'] - self.income_statements['costOfRevenue']
        self.calculated_ratios['grossProfitMargin'] = self.income_statements['grossProfit']/self.income_statements['revenue']
        self.calculated_ratios['operatingIncome_calc'] = self.income_statements['revenue'] - self.income_statements['costOfRevenue']                                                     - self.income_statements['sellingGeneralAndAdministrativeExpenses']                                                     - self.income_statements['researchAndDevelopmentExpenses']
        self.calculated_ratios['operatingProfitMargin'] = self.income_statements['operatingIncome']/self.income_statements['revenue']
#         self.calculated_ratios['incomeBeforeTax_calc'] = self.income_statements['revenue'] - self.income_statements['']
#         self.calculated_ratios['pretaxProfitMargin'] = 
        
        

    def buffet_analysis():
        '''perform analysis as per Buffetology'''
        pass

    def generate_index(self, date):
        '''
        Generates a financial index for a company based on a given date.

        Parameters:
            - date (str): A date in the format 'yyyy-mm-dd'.

        Returns:
            A string in the format '{ticker}-Q{quarter}-{year}', where ticker is the ticker of the company, quarter is the quarter 
            of the date (1-4) and year is the year of the date.
        '''
        year, month, _ = [int(i) for i in date.split('-')]
        
        if month in (1,2,3):
            quarter = 1
        elif month in (4,5,6):
            quarter = 2
        elif month in (7,8,9):
            quarter = 3
        elif month in (10, 11, 12):
            quarter = 4
        
        return f"{self._ticker}-Q{quarter}-{year}"
        


