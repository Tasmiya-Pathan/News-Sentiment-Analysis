import joblib
from flask import Flask, send_file, request, jsonify
import re
import string
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


# Download NLTK stopwords dataset while running first time
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")
# python -m spacy download en_core_web_sm
app = Flask(__name__)

model_filename = 'multinomial_nb_model.joblib'  # Load the model
clf = joblib.load(model_filename)
vectorizer_filename = 'tfidf_vectorizer.joblib'
trained_vectorizer = joblib.load(vectorizer_filename)


# Function to clean text data
def preprocess_text(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Remove special characters, URLs, and mentions
    text = re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text, flags=re.MULTILINE)

    # 4. Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])

    # 5. Tokenize the text
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatization using spaCy
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]

    # 8. Stemming using PorterStemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

    # 9. Remove short words (length <= 1)
    stemmed_tokens = [word for word in stemmed_tokens if len(word) > 1]

    # 10. Return the cleaned text
    return ' '.join(stemmed_tokens)

def text_vectorization(message):
    vectorized_msg = trained_vectorizer.transform([message])
    return vectorized_msg

def predict_sentiment(message):
    processed_msg = preprocess_text(message)
    vectorized_msg = text_vectorization(processed_msg)
    sentiment = clf.predict(vectorized_msg)
    return sentiment

# Route to predict sentiment
        #if prediction_num == 4:
         #   return "Positive"
        #else:
         #   return "Negative"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        news_text = data.get("news", "")

        if not news_text:
            return jsonify({"error": "No news text provided"}), 400

        # Predict sentiment
        sentiment = predict_sentiment(news_text)
        print(sentiment)
        # Return result
        return jsonify({"news": news_text, "sentiment": sentiment[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
