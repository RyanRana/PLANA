import unittest
from machine_learning import classify_docs_and_extract_entities

class TestDocClassificationNER(unittest.TestCase):
    def test_doc_classification_ner(self):
        docs = ["This is a financial document.", "This is a legal document."]
        result = classify_docs_and_extract_entities(docs)
        self.assertEqual(len(result), 2)
        for res in result:
            self.assertIn('document', res)
            self.assertIn('entities', res)

if __name__ == '__main__':
    unittest.main()
