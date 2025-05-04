# Voice Speaker Identification using Pretrained Whisper encoder and Contrastive Learning

The goal of this project is to develop a model capable of producing speaker embeddings — compact, fixed-length vector representations that capture the unique vocal characteristics of individual speakers. Such embeddings are useful for a range of downstream tasks, including speaker verification, clustering, and voice similarity retrieval. The core idea is that utterances from the same speaker should yield similar embeddings, while utterances from different speakers should be well-separated in the embedding space.

To train such model we use a robust pretrained audio encoder model and finetune it to capture speaker identity information using a contrastive learning approach. For the encoder model we opted to use the *Whisper Base* encoder. Whisper is a robust widely adopted transcription model developed by OpenAI. It is robust accross accents and background noise and is therefore a suitable base model.

The contrastive learning approach leverages a hard-triplet mining technique to provide the most useful training examples for the model. Additionally, a regularizing self supervised learning objective is introduced using NT-xEnt loss between embeddings resulting from augmented and original input features. Evaluation is done using metrics like top5 retrieval accuracy, intra/inter-speaker distance and equal error rate.  

TODO: Results


## Data


## Model


## Training 

## Results

## Conclusions


