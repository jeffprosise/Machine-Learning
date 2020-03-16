import xlwings as xw
import pickle, re, os, string
from sklearn.feature_extraction.text import CountVectorizer

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Data/sentiment_analysis.pkl'))
model = pickle.load(open(model_path, 'rb'))

vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Data/vocabulary.pkl'))

vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    stop_words=['i', 'we', 'you', 'the', 'and', 'am', 'are'],
    vocabulary=pickle.load(open(vocab_path, 'rb')))

table = str.maketrans('', '', string.punctuation)

def clean_text(text):
    return re.sub(' +', ' ', text.translate(table).lower())

@xw.func
def analyze_text(text):
    return model.predict_proba(vectorizer.transform([clean_text(text)]))[0][1]