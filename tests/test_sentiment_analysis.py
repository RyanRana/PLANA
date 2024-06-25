import unittest
from machine_learning import analyze_sentiment

class TestSentimentAnalysis(unittest.TestCase):
    def test_sentiment_analysis(self):
        text = "The company is performing exceptionally well."
        result = analyze_sentiment(text)
        self.assertIn('label', result[0])
        self.assertIn('score', result[0])

if __name__ == '__main__':
    unittest.main()
