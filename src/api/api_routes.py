from fastapi import FastAPI

from src.data.data_processing import VideoProcessor
from src.model.model_training import pipeline

app = FastAPI()

@app.get("/toxic")
async def get_toxic_time(video_id: str = "default text"):
    video_processor = VideoProcessor(pipeline)
    return video_processor.get_toxic_segments(video_id)
