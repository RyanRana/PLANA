import unittest
from machine_learning import analyze_financials

class TestFinancialAnalysis(unittest.TestCase):
    def test_financial_analysis(self):
        financial_data = {
            'value': [100, 200, 150, 300],
            'current_assets': [100, 150, 120, 180],
            'current_liabilities': [50, 60, 55, 70],
            'total_liabilities': [200, 220, 210, 230],
            'shareholder_equity': [400, 450, 430, 470]
        }
        result = analyze_financials(financial_data)
        self.assertIn('forecast', result)
        self.assertIn('ratios', result)
        self.assertIn('anomalies', result)

if __name__ == '__main__':
    unittest.main()
