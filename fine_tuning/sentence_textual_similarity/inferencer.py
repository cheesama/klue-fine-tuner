from fine_tuning_textual_similarity import SimilarityModel

from fastapi import FastAPI
from pydantic import BaseModel

import pytorch_lightning as pl

class Query(BaseModel):
    sentence1: str
    sentence2: str

# load checkpoint
model = None

app = FastAPI()

model = SimilarityModel.load_from_checkpoint('klue_textual_similarity.ckpt')

@app.get("/health")
def health_check():
    if model is None:
        return {'status': 400}

    return {'status': 200}

 @app.get("/inference/")
 def predict_(query: Query):
    pred = model.predict(query.sentence1, query.sentence2)

    return pred

