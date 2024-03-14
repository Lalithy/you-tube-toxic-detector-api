from fastapi import FastAPI
from youtube_transcript_api import YouTubeTranscriptApi
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


# Get Transcript with time from YouTube video
def get_transcript_with_timing(video_id):
    timed_transcript = []  # List to store timed transcript

    try:
        # Get the transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Print each word with its corresponding start and end times
        for segment in transcript:
            text = segment['text']
            start = segment['start']
            end = start + segment['duration']

            # Split the text into words
            words = text.split()

            # Calculate the duration per word
            word_duration = segment['duration'] / len(words)

            # Print each word with its start and end times
            current_time = start
            for word in words:
                # Calculate end time for the word
                word_end = current_time + word_duration
                word_time = f"[{int(current_time)}, {word_end:.0f}] - {word}"

                timed_transcript.append(word_time)
                current_time = word_end
    except Exception as e:
        print(f"An error occurred: {e}")

    return timed_transcript


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
    transcript_with_timing = get_transcript_with_timing(video_id)
    time_list = []

    for line in transcript_with_timing:
        parts = line.split("-")
        time_only = parts[0]
        word_only = parts[1]

        word_list = [word_only]
        predicted_label = pipeline.predict(word_list)
        if predicted_label[0] != "No hate and offensive speech":
            time_list.append(time_only)

    parsed_output = [eval(timestamp_str) for timestamp_str in time_list]

    # Convert the list of tuples to a list of lists
    muted_segments = [list(segment) for segment in parsed_output]

    return {"muted_segments": muted_segments}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
