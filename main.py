from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

app = FastAPI()

# Load data
df = pd.read_csv("data.csv")
df['label'] = df['class'].map(
    {0: 'Hate Speech Detected', 1: 'Offensive language detected', 2: 'No hate and offensive speech'})
df = df[['comments', 'label']]

# Data cleaning and preprocessing
stop_words = set(stopwords.words('english'))


# Clean data
def clean(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join([word for word in text.lower().split() if word not in stop_words])
    return text


df["comments"] = df["comments"].apply(clean)
df.dropna(inplace=True)

# Define pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])

# Split data
X = df["comments"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Model evaluation (optional)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


@app.get("/toxic")
async def get_toxic_time(video_id: str = "default text"):
    return {"message": video_id}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
