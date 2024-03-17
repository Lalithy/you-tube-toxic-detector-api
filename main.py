from fastapi import FastAPI, HTTPException
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
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('stopwords')

app = FastAPI()

# Constants
STOP_WORDS = set(stopwords.words('english'))


@app.get("/toxic")
async def get_toxic_time(video_id: str = "default text"):
    video_processor = VideoProcessor(pipeline)
    return video_processor.get_toxic_segments(video_id)


class VideoProcessor:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def process_transcript(self, transcript_with_timing):
        time_list = []
        for line in transcript_with_timing:
            parts = line.split("-")
            time_only = parts[0]
            word_only = parts[1]

            word_list = [word_only]
            predicted_label = self.pipeline.predict(word_list)
            if predicted_label[0] != "No hate and offensive speech":
                time_list.append(time_only)

        return [eval(timestamp_str) for timestamp_str in time_list]

    def get_toxic_segments(self, video_id):
        try:
            transcript_with_timing = get_transcript_with_timing(video_id)
            parsed_output = self.process_transcript(transcript_with_timing)

            # Convert the list of tuples to a list of lists
            muted_segments = [list(segment) for segment in parsed_output]

            return {"muted_segments": muted_segments}
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


# Load data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['label'] = df['class'].map(
            {0: 'Hate Speech Detected', 1: 'Offensive language detected', 2: 'No hate and offensive speech'})
        df = df[['comments', 'label']]

        # Data cleaning and preprocessing
        df["comments"] = df["comments"].apply(clean)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise


# Clean data
def clean(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = ' '.join([word for word in text.lower().split() if word not in STOP_WORDS])
    return text


# Define pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])


# Train model
def train_model(df):
    try:
        X = df["comments"]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        pipeline.fit(X_train, y_train)
        return pipeline, X_test, y_test
    except Exception as e:
        logger.error(f"An error occurred while training model: {e}")
        raise


# Model evaluation
def evaluate_model(pipeline, X_test, y_test):
    try:
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy}")
    except Exception as e:
        logger.error(f"An error occurred while evaluating model: {e}")
        raise


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
        logger.error(f"An error occurred while fetching transcript: {e}")
        raise

    return timed_transcript


def main():
    try:
        # Load data
        df = load_data("data.csv")

        # Train model
        pipeline, X_test, y_test = train_model(df)

        # Model evaluation
        evaluate_model(pipeline, X_test, y_test)

        # Run FastAPI app
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
