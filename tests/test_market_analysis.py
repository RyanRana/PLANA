import unittest
from machine_learning import analyze_market

class TestMarketAnalysis(unittest.TestCase):
    def test_market_analysis(self):
        texts = ["The market is growing", "The market is shrinking"]
        result = analyze_market(texts)
        self.assertIn('sentiment_scores', result)
        self.assertIn('clusters', result)
        self.assertIn('future_trends', result)

if __name__ == '__main__':
    unittest.main()
