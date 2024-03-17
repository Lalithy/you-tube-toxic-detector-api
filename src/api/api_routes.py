from fastapi import APIRouter, HTTPException

from src.data.data_processing import VideoProcessor, get_video_comments, get_sentiment
from src.model.model_training import pipeline

router = APIRouter()


# The API provides list of start seconds and end sounds on toxic connect
@router.get("/toxic")
async def get_toxic_time(video_id: str):
    try:
        video_processor = VideoProcessor(pipeline)
        return video_processor.get_toxic_segments(video_id)
    except Exception as e:
        # You can customize the error message and status code as per your requirement
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# The API provides percentage of positive, negative and neutral from the video comments analysing
@router.get("/video_sentiment")
async def analyze_video_sentiment(video_id: str):
    try:
        comments = get_video_comments(video_id)
        sentiments = [get_sentiment(comment) for comment in comments]

        total_comments = len(sentiments)
        positive_percentage = (sentiments.count('Positive') / total_comments) * 100
        negative_percentage = (sentiments.count('Negative') / total_comments) * 100
        neutral_percentage = (sentiments.count('Neutral') / total_comments) * 100

        return {
            "video_id": video_id,
            "positive": round(positive_percentage, 2),
            "negative": round(negative_percentage, 2),
            "neutral": round(neutral_percentage, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
