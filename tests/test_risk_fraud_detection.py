import unittest
from machine_learning import assess_risk_and_fraud

class TestRiskFraudDetection(unittest.TestCase):
    def test_risk_fraud_detection(self):
        financial_data = {
            'credit_score': [650, 700, 680, 720],
            'transaction_amount': [1000, 2000, 1500, 3000]
        }
        result = assess_risk_and_fraud(financial_data)
        self.assertIn('credit_scores', result)
        self.assertIn('fraud_cases', result)

if __name__ == '__main__':
    unittest.main()
