FROM tensorflow/serving

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fine_tuning_topic_classification.py .

