FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.7

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inferencer.py .
COPY fine_tuning_ner.py .


