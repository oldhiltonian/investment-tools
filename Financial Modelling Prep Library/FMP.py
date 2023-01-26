#!/usr/bin/env python
# coding: utf-8

# TODO
# - Strip the class to a .py file for easier programming and just run this notebook as a script
# - add cross-check functionality to calculate outstanding shares, eps, gross profit, operating income and compare it to the values provided in the actual statements. Assert similarity or fail stating that there is a discrepency
# - create load_financial_statements()
# - expand build_dataframe() to automatically save to excel
# - create standard_ratios() to perform all calculations, or perhaps to literally do everything start-to-finish
# - create plotting functionality

# In[3]:


import requests
import pandas as pd
from pathlib import Path


class Company:
    def __init__(self, ticker, key, data='online', period='annual', limit=120):
        self._ticker = ticker
        self.key = key
        if data == 'online':
            self.balance_sheets, self.income_statements, self.cash_flow_statements                     = self.fetch_financial_statements(ticker, period=period, limit=limit)
        else:
            '''call load_financial_statements here'''
            msg = 'not implemented yet'
            raise Exception(msg)
            
        self.balance_sheets = self.build_dataframe(self.balance_sheets)
        self.income_statements = self.build_dataframe(self.income_statements)
        self.cash_flow_statements = self.build_dataframe(self.cash_flow_statements)
        self.calculated_ratios = pd.DataFrame()
        self.cross_check()
        self.standard_ratios()


    
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

        balance_sheets = requests.get(f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{company}?period={period}&limit={limit}&apikey={self.key}')
        income_statements = requests.get(f'https://financialmodelingprep.com/api/v3/income-statement/{company}?period={period}&limit={limit}&apikey={self.key}')
        cash_flow_statements = requests.get(f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{company}?period={period}&limit={limit}&apikey={self.key}')

        return balance_sheets.json(), income_statements.json(), cash_flow_statements.json()


    def load_financial_statements(self, company, period):
        '''
        Load from a local directory
        draw distinction between annual and quartely results using an f-string'''
        pass

    
    
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
        self._check_reported_values = pd.DataFrame()
        RND_expenses = self.income_statements['researchAndDevelopmentExpenses']
        SGA_expenses = self.income_statements['sellingGeneralAndAdministrativeExpenses']
        other_expenses = self.income_statements['otherExpenses']
        revenue = self.income_statements['revenue']
        cost_of_revenue = self.income_statements['costOfRevenue']
        depreciation_amortization = self.cash_flow_statements['depreciationAndAmortization']
        interest_expense = self.income_statements['interestExpense']

        # income statement
        # ebitda, ebit
        self._check_calculated_values['ebitda'] = revenue - RND_expenses - SGA_expenses - other_expenses + depreciation_amortization

        # # #ebitdaratio, 

        # grossProfit,         # grossProfitRatio, 
        self._check_calculated_values['grossProfit'] = revenue - cost_of_revenue
        self._check_calculated_values['grossProfitRatio'] = (self._check_calculated_values['grossProfit']/revenue)*100
        
        #operatingIncome, operatingIncomeRatio
        self._check_calculated_values['operatingIncome'] = revenue - cost_of_revenue - SGA_expenses - RND_expenses
        self._check_calculated_values['operatingIncomeRatio'] = (self._check_calculated_values['operatingIncome']/revenue)*100

        # incomeBeforeTax, 
        self._check_calculated_values['incomeBeforeTax'] = self._check_calculated_values['operatingOncome'] - interest_expense
        # incomeBeforeTaxRatio, 
        self._check_calculated_values['incomeBeforeTaxRatio'] = (self._check_calculated_values['incomeBeforeTax']/revenue)*100

        # netIncome
        self._check_calculated_values['netIncome'] = 0.79*self._check_calculated_values['incomeBeforeTax']
        # netIncomeRatio, 
        self._check_calculated_values['incomeBeforeTaxRatio'] = (self._check_calculated_values['netIncome']/revenue)*100
        
        self._check_calculated_values = pd.DataFrame()
        # Calculate the above values for this df then assert that they are equal for each.
        # Actually, dont assert, but flag which ones are significantly different and then
        #     format a message to be returned in response
        return "NYE"
    
    
    def standard_ratios(self):
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
        


