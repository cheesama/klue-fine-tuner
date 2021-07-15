FROM tensorflow/serving

COPY saved_model/klue_tf_finetune_topic_classification klue_tf_finetune_topic_classification

CMD ["tensorflow_model_server", "--port=8500", "--rest_api_port=8510","--model_name=klue-tf-finetune-topic-classifier","--model_base_path=klue_tf_finetune_topic_classification"]
