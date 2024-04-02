import logging

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

STOP_WORDS = set(stopwords.words('english'))

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', DecisionTreeClassifier())
])


def train_model(df):
    try:
        x = df["comments"]
        y = df["label"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        pipeline.fit(x_train, y_train)
        return pipeline, x_test, y_test
    except Exception as e:
        logger.error(f"An error occurred while training model: {e}")
        raise


def evaluate_model(pipeline, x_test, y_test):
    try:
        y_pred = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy}")
    except Exception as e:
        logger.error(f"An error occurred while evaluating model: {e}")
        raise
