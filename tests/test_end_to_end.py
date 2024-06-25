import unittest
import requests

class TestEndToEnd(unittest.TestCase):
    def test_upload_plan(self):
        url = 'http://localhost:5000/upload'
        data = {
            'name': 'Test Plan',
            'content': 'This is a test business plan.',
            'upload_date': '2024-01-01'
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 201)

    def test_analyze_sentiment(self):
        url = 'http://localhost:5000/analyze_sentiment'
        data = {
            'plan_id': 1,
            'content': 'The company is performing exceptionally well.',
            'timestamp': '2024-01-01T00:00:00Z'
        }
        response = requests.post(url, json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('sentiment_score', response.json())

    # Similarly add tests for other endpoints

if __name__ == '__main__':
    unittest.main()
