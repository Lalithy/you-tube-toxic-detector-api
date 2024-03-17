from fastapi import APIRouter

from src.data.data_processing import VideoProcessor
from src.model.model_training import pipeline

router = APIRouter()


# The API provides list of start seconds and end sounds on toxic connect
@router.get("/toxic")
async def get_toxic_time(video_id: str = "default text"):
    video_processor = VideoProcessor(pipeline)
    return video_processor.get_toxic_segments(video_id)
