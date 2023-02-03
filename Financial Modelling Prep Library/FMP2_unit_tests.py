import unittest
from FMP2 import FinancialData

class TestFinancialStatements(unittest.TestCase):
    def setUp(self, ticker, api_key='', data='local', period='annual', limit=120):
        self.ticker = ticker.upper().strip()
        self.api_key = str(api_key)
        self.data = data.lower().strip()
        self.period = period.lower.strip()
        self.limit = int(limit)
    
    def test_length_of_statements(self):
        '''returned frames cannot have a length of zero
            and all must have equal lengths'''
        fs = FinancialData(self.ticker, self.api_key, self.data, self.period, self.limit)
        self.assertNotEqual(len(fs.balance_sheets), 0)
        self.assertNotEqual(len(fs.income_statements), 0)
        self.assertNotEqual(len(fs.cash_flow_statements), 0)
        self.assertEqual(len(fs.balance_sheets), len(fs.income_statements))
        self.assertEqual(len(fs.balance_sheets), len(fs.cash_flow_statements))

    def test_matching_index(self):
        '''Frames must have the same index to facilitate correct calculations of financial ratios'''
        fs = FinancialData(self.ticker, self.api_key, self.data, self.period, self.limit)
        self.assertEqual(fs.balance_sheets.index.tolist(), fs.income_statements.index.tolist())
        self.assertEqual(fs.balance_sheets.index.tolist(), fs.cash_flow_statements.index.tolist())