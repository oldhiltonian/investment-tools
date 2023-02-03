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
        self._frame_indecies = common_elements
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