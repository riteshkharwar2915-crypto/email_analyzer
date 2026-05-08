from flask import Flask, request, render_template_string, jsonify
import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import joblib
import threading
import time

app = Flask(__name__)

class SafeEmailPredictor:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        self.spam_keywords = ['free', 'win', 'prize', 'urgent', 'click', 'limited', 'million', 'viagra', 'casino']
        self.phishing_keywords = ['verify', 'account', 'password', 'security', 'suspended', 'billing', 'bank', 'login']
        self.safe_keywords = ['meeting', 'project', 'invoice', 'payment', 'schedule', 'team', 'update', 'confirm']
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9s]', ' ', text)
        text = re.sub(r's+', ' ', text).strip()
        return text
    
    def rule_based_category(self, text):
        text_lower = text.lower()
        spam_score = sum(1 for word in self.spam_keywords if word in text_lower)
        phishing_score = sum(1 for word in self.phishing_keywords if word in text_lower)
        safe_score = sum(1 for word in self.safe_keywords if word in text_lower)
        
        if spam_score >= 2 or '!!!' in text_lower:
            return "SPAM"
        elif phishing_score >= 2 or any(word in text_lower for word in ['http', 'www', 'link']):
            return "PHISHING"
        elif safe_score >= 1:
            return "SAFE"
        else:
            return "UNKNOWN"
    
    def safety_score(self, text):
        text_lower = text.lower()
        score = 100
        
        spam_count = sum(1 for word in self.spam_keywords if word in text_lower)
        phishing_count = sum(1 for word in self.phishing_keywords if word in text_lower)
        
        score -= spam_count * 25
        score -= phishing_count * 30
        score -= min(text_lower.count('!'), 10) * 8
        score -= min(text_lower.count('$'), 3) * 20
        
        if re.search(r'http[s]?://|www.|.com/|.net/', text_lower):
            score -= 25
        
        safe_count = sum(1 for word in self.safe_keywords if word in text_lower)
        score += safe_count * 15
        
        return max(0, min(100, score))
    
    def predict(self, email_text):
        clean_text = self.preprocess_text(email_text)
        
        ml_category = "RULE_BASED"
        ml_confidence = 0
        try:
            if not self.is_trained:
                self._train_model()
            
            pred = self.pipeline.predict([clean_text])[0]
            ml_category = self.label_encoder.inverse_transform([pred])[0]
            ml_confidence = np.max(self.pipeline.predict_proba([clean_text])[0])
        except Exception as e:
            ml_category = "RULE_BASED"
        
        rule_category = self.rule_based_category(email_text)
        safety = self.safety_score(email_text)
        final_category = ml_category if ml_confidence > 0.6 else rule_category
        
        return {
            'category': final_category,
            'ml_category': ml_category,
            'ml_confidence': f"{ml_confidence:.0%}",
            'rule_category': rule_category,
            'safety_score': safety,
            'is_safe': safety > 75,
            'scan_details': self._get_scan_details(email_text)
        }
    
    def _get_scan_details(self, text):
        text_lower = text.lower()
        details = []
        
        if text_lower.count('!') > 2:
            details.append(f"{text_lower.count('!')} EXCLAMATION MARKS")
        
        urls = re.findall(r'http[s]?://|www.|.com', text_lower)
        if urls:
            details.append(f"{len(urls)} SUSPICIOUS LINKS")
        
        spam_words = [w for w in self.spam_keywords if w in text_lower]
        if spam_words:
            details.append(f"SPAM: {', '.join(spam_words[:3])}")
        
        safe_words = [w for w in self.safe_keywords if w in text_lower]
        if safe_words:
            details.append(f"SAFE: {', '.join(safe_words[:3])}")
        
        return details
    
    def _train_model(self):
        training_data = [
            "team meeting tomorrow", "invoice payment received", "project update", "schedule change",
            "lunch meeting", "status update", "confirm attendance",
            "free viagra", "win million dollars", "casino bonus", "free iphone",
            "click here now", "limited time offer", "urgent prize",
            "account suspended verify", "security alert login", "bank issue update",
            "password reset click", "billing information required", "confirm details now"
        ]
        
        labels = ['safe']*7 + ['spam']*7 + ['phishing']*6
        
        self.label_encoder.fit(labels)
        self.pipeline = make_pipeline(
            TfidfVectorizer(max_features=1000, stop_words='english'),
            RandomForestClassifier(n_estimators=50, random_state=42)
        )
        
        self.pipeline.fit(training_data, self.label_encoder.transform(labels))
        self.is_trained = True

# Global predictor
predictor = SafeEmailPredictor()

# HTML Frontend Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🛡️ SAFE EMAIL PREDICTOR</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        
        .container { max-width: 900px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden; }
        
        .header { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 30px; text-align: center; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        
        .input-section { padding: 40px; background: #f8f9fa; }
        textarea { width: 100%; height: 120px; padding: 20px; border: 2px solid #e9ecef; border-radius: 12px; font-size: 16px; resize: vertical; 
                   transition: border-color 0.3s; }
        textarea:focus { outline: none; border-color: #4facfe; box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1); }
        .analyze-btn { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; 
                       padding: 15px 40px; font-size: 18px; border-radius: 12px; cursor: pointer; margin-top: 20px;
                       transition: transform 0.2s; }
        .analyze-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3); }
        .analyze-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #4facfe; border-radius: 50%; width: 40px; height: 40px; 
                   animation: spin 1s linear infinite; margin: 0 auto 10px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .results { padding: 40px; display: none; }
        .safety-score { text-align: center; margin-bottom: 30px; }
        .score-circle { width: 150px; height: 150px; border-radius: 50%; margin: 0 auto 20px; display: flex; align-items: center; 
                        justify-content: center; font-size: 2.5em; font-weight: bold; position: relative; }
        .score-safe { background: conic-gradient(#4ade80 0deg, #4ade80 75%, #fbbf24 75deg 90%, #f87171 90deg 360deg); }
        .score-caution { background: conic-gradient(#fbbf24 0deg, #fbbf24 60%, #f87171 60deg 360deg); }
        .score-danger { background: conic-gradient(#f87171 0deg 100%, #dc2626 100deg 360deg); }
        .score-circle span { background: white; width: 110px; height: 110px; border-radius: 50%; display: flex; align-items: center; 
                            justify-content: center; font-size: 1.2em; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        
        .category-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .category-card { padding: 25px; border-radius: 15px; text-align: center; font-weight: bold; font-size: 1.1em; }
        .cat-safe { background: linear-gradient(135deg, #4ade80, #22c55e); color: white; }
        .cat-spam { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: white; }
        .cat-phishing { background: linear-gradient(135deg, #f87171, #ef4444); color: white; }
        
        .details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px; }
        .detail-card { background: #f8f9fa; padding: 25px; border-radius: 15px; }
        .detail-card h3 { color: #4facfe; margin-bottom: 15px; font-size: 1.2em; }
        .detail-list { list-style: none; }
        .detail-list li { padding: 8px 0; border-bottom: 1px solid #e9ecef; }
        .detail-list li:last-child { border-bottom: none; }
        
        .recommendation { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 25px; 
                         border-radius: 15px; text-align: center; font-size: 1.3em; font-weight: bold; margin-top: 20px; }
        
        @media (max-width: 768px) { .details-grid { grid-template-columns: 1fr; } .category-grid { grid-template-columns: repeat(2, 1fr); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ SAFE EMAIL PREDICTOR</h1>
            <p>AI-Powered Spam & Phishing Detection • ML + Rules Engine</p>
        </div>
        
        <div class="input-section">
            <textarea id="emailText" placeholder="Paste your email content here...&#10;&#10;Example: 'URGENT!!! FREE $1M WINNER! Click: http://scam.com'"></textarea>
            <br>
            <button class="analyze-btn" onclick="analyzeEmail()">🔍 ANALYZE EMAIL</button>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing with AI...</p>
            </div>
        </div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        function analyzeEmail() {
            const emailText = document.getElementById('emailText').value.trim();
            if (!emailText) return alert('Please enter email content!');
            
            document.querySelector('.analyze-btn').disabled = true;
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({email: emailText})
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            })
            .catch(error => {
                alert('Analysis failed! ' + error);
            })
            .finally(() => {
                document.querySelector('.analyze-btn').disabled = false;
                document.getElementById('loading').style.display = 'none';
            });
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            const score = data.safety_score;
            let scoreClass = 'score-danger';
            let scoreText = 'DANGER';
            let recText = '⛔ DELETE IMMEDIATELY';
            
            if (score > 75) {
                scoreClass = 'score-safe';
                scoreText = 'SAFE';
                recText = '✅ Open normally';
            } else if (score > 40) {
                scoreClass = 'score-caution';
                scoreText = 'CAUTION';
                recText = '⚠️ Check sender & links';
            }
            
            const catClass = data.category.toLowerCase().replace(' ', '-');
            
            resultsDiv.innerHTML = `
                <div class="safety-score">
                    <div class="score-circle ${scoreClass}">
                        <span>${score}/100</span>
                    </div>
                    <h2>${scoreText}</h2>
                </div>
                
                <div class="category-grid">
                    <div class="category-card cat-${catClass}">
                        🎯 FINAL: ${data.category}
                    </div>
                    <div class="category-card cat-${data.ml_category.toLowerCase().replace(' ', '-')}">
                        🤖 ML: ${data.ml_category} (${data.ml_confidence})
                    </div>
                    <div class="category-card cat-${data.rule_category.toLowerCase().replace(' ', '-')}">
                        ⚖️ Rules: ${data.rule_category}
                    </div>
                </div>
                
                <div class="recommendation">${recText}</div>
                
                <div class="details-grid">
                    <div class="detail-card">
                        <h3>🔎 Quick Scan</h3>
                        <ul class="detail-list">
                            ${data.scan_details.map(detail => `<li>• ${detail}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="detail-card">
                        <h3>📊 Model Confidence</h3>
                        <ul class="detail-list">
                            <li>🔬 ML Confidence: ${data.ml_confidence}</li>
                            <li>✅ Rule Backup: Always active</li>
                            <li>🎯 Safety Threshold: ${data.is_safe ? 'PASSED' : 'FAILED'}</li>
                        </ul>
                    </div>
                </div>
            `;
            
            resultsDiv.style.display = 'block';
            resultsDiv.scrollIntoView({behavior: 'smooth'});
        }
        
        // Test emails
        document.getElementById('emailText').value = `URGENT!!! YOUR ACCOUNT SUSPENDED!!! 
FREE $1,000,000 WINNER!!! CLICK HERE NOW: http://free-money-scam.com/claim`;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email', '')
    result = predictor.predict(email_text)
    return jsonify(result)

if __name__ == '__main__':
    print("🚀 Starting Safe Email Predictor Web App...")
    print("🌐 Open: http://localhost:5000")
    print("📱 Works on mobile too!")
    app.run(debug=True, host='0.0.0.0', port=5000)
