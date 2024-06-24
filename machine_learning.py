from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
import numpy as np
import spacy
import openai

# Sentiment Analysis using BERT
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return result

# Market Analysis using Text Clustering and Forecasting
def analyze_market(texts):
    # Sentiment Analysis
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment_scores = [sentiment_pipeline(text) for text in texts]
    
    # Market Segmentation using Clustering
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    clusters = kmeans.labels_
    
    # Trend Forecasting (Dummy implementation)
    historical_data = np.random.rand(100, 1)  # Replace with actual data
    model = LinearRegression()
    model.fit(np.arange(100).reshape(-1, 1), historical_data)
    future_trends = model.predict(np.array([101, 102, 103]).reshape(-1, 1))

    return {
        "sentiment_scores": sentiment_scores,
        "clusters": clusters.tolist(),
        "future_trends": future_trends.tolist()
    }

# Financial Projections
def analyze_financials(financial_data):
    df = pd.DataFrame(financial_data)
    model = ARIMA(df['value'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=5)
    
    ratios = {
        "current_ratio": df['current_assets'].sum() / df['current_liabilities'].sum(),
        "debt_to_equity": df['total_liabilities'].sum() / df['shareholder_equity'].sum()
    }
    
    anomalies = df[df['value'] > df['value'].mean() + 3 * df['value'].std()]
    
    return {
        "forecast": forecast.tolist(),
        "ratios": ratios,
        "anomalies": anomalies.to_dict()
    }

# Credit Risk and Fraud Detection
def assess_risk_and_fraud(financial_data):
    df = pd.DataFrame(financial_data)
    
    credit_scores = df['credit_score'].mean()  # Replace with actual model
    
    model = IsolationForest()
    model.fit(df[['transaction_amount']])
    anomalies = model.predict(df[['transaction_amount']])
    
    return {
        "credit_scores": credit_scores,
        "fraud_cases": anomalies.tolist()
    }

# Document Classification and Named Entity Recognition
def classify_docs_and_extract_entities(docs):
    nlp = spacy.load("en_core_web_sm")
    results = []
    for doc in docs:
        doc_nlp = nlp(doc)
        entities = [(ent.text, ent.label_) for ent in doc_nlp.ents]
        results.append({
            "document": doc,
            "entities": entities
        })
    return results

# Weighted Averaging and GPT-3 Feedback Generation
def generate_feedback(results):
    weights = {
        'sentiment': 0.2,
        'market': 0.3,
        'financial': 0.3,
        'risk_fraud': 0.1,
        'classification_ner': 0.1
    }
    
    normalized_scores = {}
    for key, value in results.items():
        normalized_scores[key] = (value - min(value)) / (max(value) - min(value))
    
    final_grade = sum(normalized_scores[key] * weights[key] for key in weights)
    
    feedback_text = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Provide a comprehensive feedback based on the following analysis results: {results}",
        max_tokens=150
    ).choices[0].text.strip()
    
    return {
        "final_grade": final_grade,
        "feedback_text": feedback_text
    }
