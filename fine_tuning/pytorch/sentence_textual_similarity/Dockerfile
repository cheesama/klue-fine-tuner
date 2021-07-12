FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.7

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inferencer.py .
COPY fine_tuning_textual_similarity.py .
COPY klue_textual_similarity.ckpt .

CMD ["uvicorn","inferencer:app","--reload","--host","0.0.0.0","--port","8000"]


