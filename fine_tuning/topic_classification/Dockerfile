FROM pytorchlightning/pytorch_lightning

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inferencer.py .
COPY fine_tuning_topic_classification.py .


