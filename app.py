from flask import Flask, request, render_template
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data if not already available
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load saved models
rf_model = joblib.load('/Users/muhammadalizaffar/Developers_hub/Task 3/fake_news_rf.pkl')
tokenizer = joblib.load('/Users/muhammadalizaffar/Developers_hub/Task 3/tokenizer.pkl')
lstm_model = load_model('/Users/muhammadalizaffar/Developers_hub/Task 3/fake_news_lstm.h5')

# Define a cleaning function (similar to training)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    article = request.form.get('article')
    if not article:
        return render_template('index.html', prediction_text='Please enter a news article!')
    
    # Clean the article
    cleaned_article = clean_text(article)
    
    # Predict using the Random Forest model
    rf_prediction = rf_model.predict([cleaned_article])[0]
    
    # For LSTM prediction, tokenize and pad the sequence
    seq = tokenizer.texts_to_sequences([cleaned_article])
    pad_seq = pad_sequences(seq, maxlen=150)
    lstm_pred_prob = lstm_model.predict(pad_seq)[0][0]
    lstm_prediction = "REAL" if lstm_pred_prob > 0.5 else "FAKE"
    
    result = (f"Random Forest Prediction: {rf_prediction} | "
              f"LSTM Prediction: {lstm_prediction} (Prob: {lstm_pred_prob:.2f})")
    
    return render_template('index.html', prediction_text=result, article=article)

if __name__ == '__main__':
    app.run(debug=True)
