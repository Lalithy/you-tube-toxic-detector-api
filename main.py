import logging
import uvicorn
from src.model.model_training import train_model, evaluate_model
from src.utils.helpers import clean
from src.api.fastapi_routes import app
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def main():
    try:
        # Load data
        df = load_data("src/data/data.csv")

        # Train model
        pipeline, X_test, y_test = train_model(df)

        # Model evaluation
        evaluate_model(pipeline, X_test, y_test)

        # Run FastAPI app
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
