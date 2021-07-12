from fine_tuning_topic_classification import TopicModel

from fastapi import FastAPI

# load checkpoint
model = None

app = FastAPI()

model = TopicModel.load_from_checkpoint('klue_topic_classification.ckpt')

@app.get("/health")
def health_check():
    if model is None:
        return {'status': 400}

    return {'status': 200}

@app.get("/inference/{query}")
def predict_topic(query: str):
    pred = model.predict(query)



