from flask import Flask, request, render_template_string, jsonify
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Analytics
history = []
total_visitors = 0

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
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
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

        except:
            ml_category = "RULE_BASED"

        rule_category = self.rule_based_category(email_text)
        safety = self.safety_score(email_text)

        final_category = ml_category.upper() if ml_confidence > 0.6 else rule_category

        return {
            'category': final_category,
            'ml_category': ml_category.upper(),
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
            "team meeting tomorrow",
            "invoice payment received",
            "project update",
            "schedule change",
            "free viagra",
            "win million dollars",
            "casino bonus",
            "account suspended verify",
            "security alert login",
            "password reset click"
        ]

        labels = [
            'safe',
            'safe',
            'safe',
            'safe',
            'spam',
            'spam',
            'spam',
            'phishing',
            'phishing',
            'phishing'
        ]

        self.label_encoder.fit(labels)

        self.pipeline = make_pipeline(
            TfidfVectorizer(max_features=1000, stop_words='english'),
            RandomForestClassifier(n_estimators=50, random_state=42)
        )

        self.pipeline.fit(
            training_data,
            self.label_encoder.transform(labels)
        )

        self.is_trained = True

predictor = SafeEmailPredictor()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>SAFE EMAIL PREDICTOR</title>

<meta name="viewport" content="width=device-width, initial-scale=1">

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
body{
background:linear-gradient(135deg,#667eea,#764ba2);
font-family:Arial;
padding:20px;
color:white;
}

.container{
max-width:900px;
margin:auto;
background:white;
color:black;
border-radius:20px;
padding:30px;
}

textarea{
width:100%;
height:140px;
padding:15px;
border-radius:10px;
font-size:16px;
}

button{
margin-top:20px;
padding:15px 30px;
border:none;
background:#4facfe;
color:white;
font-size:18px;
border-radius:10px;
cursor:pointer;
}

.results{
margin-top:30px;
display:none;
}

.analytics{
margin-top:50px;
text-align:center;
}

canvas{
max-width:400px;
margin:auto;
}
</style>
</head>

<body>

<div class="container">

<h1>SAFE EMAIL PREDICTOR</h1>

<textarea id="emailText"></textarea>

<br>

<button onclick="analyzeEmail()">ANALYZE EMAIL</button>

<div class="results" id="results"></div>

<div class="analytics">

<h2>WEBSITE ANALYTICS</h2>

<canvas id="analyticsChart"></canvas>

<h3>Total Visitors: <span id="visitorCount">0</span></h3>

</div>

</div>

<script>

function analyzeEmail(){

const emailText=document.getElementById('emailText').value;

fetch('/predict',{
method:'POST',
headers:{
'Content-Type':'application/json'
},
body:JSON.stringify({
email:emailText
})
})
.then(response=>response.json())
.then(data=>{

document.getElementById('results').style.display='block';

document.getElementById('results').innerHTML=`
<h2>FINAL CATEGORY: ${data.category}</h2>
<h3>Safety Score: ${data.safety_score}/100</h3>
<p>ML Confidence: ${data.ml_confidence}</p>
`;

loadAnalytics();

});
}

function loadAnalytics(){

fetch('/analytics')
.then(response=>response.json())
.then(data=>{

document.getElementById('visitorCount').innerText=data.total_visitors;

const ctx=document.getElementById('analyticsChart');

new Chart(ctx,{
type:'pie',
data:{
labels:['SAFE','SPAM','PHISHING'],
datasets:[{
data:[
data.safe,
data.spam,
data.phishing
],
backgroundColor:[
'#22c55e',
'#f59e0b',
'#ef4444'
]
}]
}
});

});

}

loadAnalytics();

</script>

</body>
</html>
"""

@app.route('/')
def index():
    global total_visitors
    total_visitors += 1
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    email_text = data.get('email', '')

    result = predictor.predict(email_text)

    history.append(result['category'])

    return jsonify(result)

@app.route('/analytics')
def analytics():

    return jsonify({
        'safe': history.count('SAFE'),
        'spam': history.count('SPAM'),
        'phishing': history.count('PHISHING'),
        'total_visitors': total_visitors
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
