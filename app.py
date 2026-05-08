from flask import Flask, request, render_template_string, jsonify
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Analytics Storage
history = []
total_visitors = 0

class SafeEmailPredictor:

    def __init__(self):

        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False

        self.spam_keywords = [
            'free', 'win', 'prize', 'urgent',
            'click', 'limited', 'million',
            'viagra', 'casino'
        ]

        self.phishing_keywords = [
            'verify', 'account', 'password',
            'security', 'suspended',
            'billing', 'bank', 'login'
        ]

        self.safe_keywords = [
            'meeting', 'project', 'invoice',
            'payment', 'schedule', 'team',
            'update', 'confirm'
        ]

    def preprocess_text(self, text):

        text = text.lower()

        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def rule_based_category(self, text):

        text_lower = text.lower()

        spam_score = sum(
            1 for word in self.spam_keywords
            if word in text_lower
        )

        phishing_score = sum(
            1 for word in self.phishing_keywords
            if word in text_lower
        )

        safe_score = sum(
            1 for word in self.safe_keywords
            if word in text_lower
        )

        if spam_score >= 2 or '!!!' in text_lower:
            return "SPAM"

        elif phishing_score >= 2 or any(
            word in text_lower
            for word in ['http', 'www', 'link']
        ):
            return "PHISHING"

        elif safe_score >= 1:
            return "SAFE"

        else:
            return "UNKNOWN"

    def safety_score(self, text):

        text_lower = text.lower()

        score = 100

        spam_count = sum(
            1 for word in self.spam_keywords
            if word in text_lower
        )

        phishing_count = sum(
            1 for word in self.phishing_keywords
            if word in text_lower
        )

        score -= spam_count * 25
        score -= phishing_count * 30
        score -= min(text_lower.count('!'), 10) * 8
        score -= min(text_lower.count('$'), 3) * 20

        if re.search(
            r'http[s]?://|www.|.com/|.net/',
            text_lower
        ):
            score -= 25

        safe_count = sum(
            1 for word in self.safe_keywords
            if word in text_lower
        )

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

            ml_category = self.label_encoder.inverse_transform(
                [pred]
            )[0]

            ml_confidence = np.max(
                self.pipeline.predict_proba(
                    [clean_text]
                )[0]
            )

        except:
            ml_category = "RULE_BASED"

        rule_category = self.rule_based_category(email_text)

        safety = self.safety_score(email_text)

        final_category = (
            ml_category.upper()
            if ml_confidence > 0.6
            else rule_category
        )

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

            details.append(
                f"{text_lower.count('!')} EXCLAMATION MARKS"
            )

        urls = re.findall(
            r'http[s]?://|www.|.com',
            text_lower
        )

        if urls:

            details.append(
                f"{len(urls)} SUSPICIOUS LINKS"
            )

        spam_words = [
            w for w in self.spam_keywords
            if w in text_lower
        ]

        if spam_words:

            details.append(
                f"SPAM: {', '.join(spam_words[:3])}"
            )

        safe_words = [
            w for w in self.safe_keywords
            if w in text_lower
        ]

        if safe_words:

            details.append(
                f"SAFE: {', '.join(safe_words[:3])}"
            )

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

            TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            ),

            RandomForestClassifier(
                n_estimators=50,
                random_state=42
            )
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

<meta name="viewport"
content="width=device-width, initial-scale=1">

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Segoe UI',sans-serif;
}

body{
background:linear-gradient(135deg,#667eea,#764ba2);
min-height:100vh;
padding:30px;
}

.container{
max-width:1000px;
margin:auto;
background:rgba(255,255,255,0.15);
backdrop-filter:blur(15px);
border-radius:25px;
padding:40px;
box-shadow:0 8px 32px rgba(0,0,0,0.2);
color:white;
}

h1{
text-align:center;
font-size:42px;
margin-bottom:10px;
}

.subtitle{
text-align:center;
margin-bottom:40px;
opacity:0.8;
}

textarea{
width:100%;
height:180px;
border:none;
outline:none;
padding:20px;
border-radius:20px;
font-size:16px;
background:rgba(255,255,255,0.2);
color:white;
resize:none;
}

textarea::placeholder{
color:#ddd;
}

button{
margin-top:25px;
width:100%;
padding:18px;
border:none;
border-radius:18px;
font-size:20px;
font-weight:bold;
cursor:pointer;
background:linear-gradient(135deg,#00c6ff,#0072ff);
color:white;
transition:0.3s;
}

button:hover{
transform:scale(1.02);
}

.results{
margin-top:40px;
padding:30px;
border-radius:20px;
background:rgba(255,255,255,0.1);
display:none;
}

.result-box{
text-align:center;
margin-bottom:20px;
}

.score{
font-size:55px;
font-weight:bold;
margin:15px 0;
}

.analytics{
margin-top:50px;
padding:30px;
border-radius:20px;
background:rgba(255,255,255,0.1);
}

.analytics h2{
text-align:center;
margin-bottom:30px;
font-size:32px;
}

.chart-container{
width:100%;
max-width:420px;
margin:auto;
}

.visitors{
text-align:center;
margin-top:25px;
font-size:24px;
font-weight:bold;
}

.stats-grid{
display:grid;
grid-template-columns:
repeat(auto-fit,minmax(180px,1fr));
gap:20px;
margin-top:35px;
}

.stat-card{
background:rgba(255,255,255,0.12);
padding:20px;
border-radius:20px;
text-align:center;
}

.stat-card h3{
font-size:18px;
margin-bottom:10px;
}

.stat-card p{
font-size:28px;
font-weight:bold;
}

.safe{
color:#22c55e;
}

.spam{
color:#f59e0b;
}

.phishing{
color:#ef4444;
}

</style>

</head>

<body>

<div class="container">

<h1>🛡 SAFE EMAIL PREDICTOR</h1>

<p class="subtitle">
AI Powered Spam & Phishing Detection
</p>

<textarea id="emailText"
placeholder="Paste your email content here..."></textarea>

<button onclick="analyzeEmail()">
🔍 ANALYZE EMAIL
</button>

<div class="results" id="results"></div>

<div class="analytics">

<h2>📊 WEBSITE ANALYTICS</h2>

<div class="chart-container">

<canvas id="analyticsChart"></canvas>

</div>

<div class="stats-grid">

<div class="stat-card">

<h3>🟢 SAFE</h3>

<p class="safe" id="safeCount">0</p>

</div>

<div class="stat-card">

<h3>🟠 SPAM</h3>

<p class="spam" id="spamCount">0</p>

</div>

<div class="stat-card">

<h3>🔴 PHISHING</h3>

<p class="phishing" id="phishingCount">0</p>

</div>

</div>

<div class="visitors">

👥 Total Visitors:
<span id="visitorCount">0</span>

</div>

</div>

</div>

<script>

let chart;

function analyzeEmail(){

const emailText=
document.getElementById('emailText').value;

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

document.getElementById(
'results'
).style.display='block';

document.getElementById(
'results'
).innerHTML=`

<div class="result-box">

<h2>FINAL CATEGORY</h2>

<div class="score">
${data.category}
</div>

<h3>
Safety Score:
${data.safety_score}/100
</h3>

<br>

<p>
🤖 ML Confidence:
${data.ml_confidence}
</p>

</div>

`;

loadAnalytics();

});

}

function loadAnalytics(){

fetch('/analytics')

.then(response=>response.json())

.then(data=>{

document.getElementById(
'visitorCount'
).innerText=data.total_visitors;

document.getElementById(
'safeCount'
).innerText=data.safe;

document.getElementById(
'spamCount'
).innerText=data.spam;

document.getElementById(
'phishingCount'
).innerText=data.phishing;

const ctx=document.getElementById(
'analyticsChart'
);

if(chart){
chart.destroy();
}

chart=new Chart(ctx,{

type:'pie',

data:{

labels:[
'SAFE',
'SPAM',
'PHISHING'
],

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
],

borderWidth:2

}]

},

options:{

responsive:true,

plugins:{
legend:{
labels:{
color:'white',
font:{
size:16
}
}
}
}

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

    return render_template_string(
        HTML_TEMPLATE
    )

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

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
