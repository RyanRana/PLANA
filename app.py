
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from machine_learning import analyze_sentiment, analyze_market, analyze_financials, assess_risk_and_fraud, classify_docs_and_extract_entities, generate_feedback

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///business_plans.db'
db = SQLAlchemy(app)

class BusinessPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    upload_date = db.Column(db.Date, nullable=False)

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('business_plan.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

class FinalFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('business_plan.id'), nullable=False)
    final_grade = db.Column(db.Float, nullable=False)
    feedback_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

@app.route('/upload', methods=['POST'])
def upload_plan():
    data = request.get_json()
    new_plan = BusinessPlan(name=data['name'], content=data['content'], upload_date=data['upload_date'])
    db.session.add(new_plan)
    db.session.commit()
    return jsonify({"message": "Business plan uploaded successfully"}), 201

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_route():
    data = request.get_json()
    sentiment_score = analyze_sentiment(data['content'])
    new_result = AnalysisResult(plan_id=data['plan_id'], analysis_type='sentiment', result=str(sentiment_score), timestamp=data['timestamp'])
    db.session.add(new_result)
    db.session.commit()
    return jsonify({"sentiment_score": sentiment_score}), 200

@app.route('/analyze_market', methods=['POST'])
def analyze_market_route():
    data = request.get_json()
    market_results = analyze_market(data['content'])
    new_result = AnalysisResult(plan_id=data['plan_id'], analysis_type='market', result=market_results, timestamp=data['timestamp'])
    db.session.add(new_result)
    db.session.commit()
    return jsonify({"market_analysis": market_results}), 200

@app.route('/analyze_financials', methods=['POST'])
def analyze_financials_route():
    data = request.get_json()
    financial_results = analyze_financials(data['financial_data'])
    new_result = AnalysisResult(plan_id=data['plan_id'], analysis_type='financial', result=financial_results, timestamp=data['timestamp'])
    db.session.add(new_result)
    db.session.commit()
    return jsonify({"financial_analysis": financial_results}), 200

@app.route('/assess_risk_and_fraud', methods=['POST'])
def assess_risk_and_fraud_route():
    data = request.get_json()
    risk_fraud_results = assess_risk_and_fraud(data['financial_data'])
    new_result = AnalysisResult(plan_id=data['plan_id'], analysis_type='risk_fraud', result=risk_fraud_results, timestamp=data['timestamp'])
    db.session.add(new_result)
    db.session.commit()
    return jsonify({"risk_fraud_analysis": risk_fraud_results}), 200

@app.route('/classify_docs_and_extract_entities', methods=['POST'])
def classify_docs_and_extract_entities_route():
    data = request.get_json()
    classification_ner_results = classify_docs_and_extract_entities(data['documents'])
    new_result = AnalysisResult(plan_id=data['plan_id'], analysis_type='classification_ner', result=classification_ner_results, timestamp=data['timestamp'])
    db.session.add(new_result)
    db.session.commit()
    return jsonify({"classification_ner_analysis": classification_ner_results}), 200

@app.route('/final_feedback', methods=['POST'])
def final_feedback_route():
    data = request.get_json()
    results = data['results']  # This should be a dictionary of model results
    feedback = generate_feedback(results)
    new_feedback = FinalFeedback(plan_id=data['plan_id'], final_grade=feedback['final_grade'], feedback_text=feedback['feedback_text'], timestamp=data['timestamp'])
    db.session.add(new_feedback)
    db.session.commit()
    return jsonify({"final_grade": feedback['final_grade'], "feedback_text": feedback['feedback_text']}), 200

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
