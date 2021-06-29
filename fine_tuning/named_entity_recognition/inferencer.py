from fine_tuning_ner import NERModel

import fastapi
import pytorch_lightning as pl

# load checkpoint
model = None

app = FastAPI()

model = NERModel.load_from_checkpoint('klue_ner.ckpt')

@app.get("/health")
def health_check():
    if model is None:
        return {'status': 400}

    return {'status': 200}

@app.get("/inference/{query}")
def predict_ner(query: str):
    pred = model.predict(query)

    return pred



