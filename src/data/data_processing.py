from fastapi import HTTPException
from youtube_transcript_api import YouTubeTranscriptApi
import logging
from googleapiclient.discovery import build
from langdetect import detect
from textblob import TextBlob
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# YouTube API Key
API_KEY = os.environ.get('YOUTUBE_API_KEY')
if not API_KEY:
    raise ValueError("YouTube API key not found in environment variables")

# Initialize YouTube Data API Service
youtube = build('youtube', 'v3', developerKey=API_KEY)


# The method provide list of comment in the video from the video id
def get_video_comments(video_id):
    comments = []
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText'
    ).execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        # Check if the comment is not empty and contains visible characters
        if comment.strip() and comment.strip().replace(' ', ''):
            try:
                # Attempt language detection
                if detect(comment) == 'en':
                    comments.append(comment)
            except:
                pass  # Skip comments that cause language detection errors

    return comments


# The method provide sentiment status
def get_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'


# The method provide list of transcript from the video
def get_transcript_with_timing(video_id):
    timed_transcript = []  # List to store timed transcript

    try:
        # Get the transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        for segment in transcript:
            text = segment['text']
            start = segment['start']

            # Split the text into words
            words = text.split()

            # Calculate the duration per word
            word_duration = segment['duration'] / len(words)

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


# The is class do process of the toxic detect and analyze
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
