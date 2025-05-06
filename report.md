# Voice Speaker Identification using Pretrained Whisper encoder and Contrastive Learning

The goal of this project is to develop a model capable of producing speaker embeddings — compact, fixed-length vector representations that capture the unique vocal characteristics of individual speakers. Such embeddings are useful for a range of downstream tasks, including speaker verification, clustering, and voice similarity retrieval. The core idea is that utterances from the same speaker should yield similar embeddings, while utterances from different speakers should be well-separated in the embedding space.

To train such model we use a robust pretrained audio encoder model and finetune it to capture speaker identity information using a contrastive learning approach. For the encoder model we opted to use the *Whisper Base* encoder. Whisper is a robust widely adopted transcription model developed by OpenAI. It is robust accross accents and background noise and is therefore a suitable base model.

The contrastive learning approach leverages a hard-triplet mining technique to provide the most useful training examples for the model. Additionally, a regularizing self supervised learning objective is introduced using NT-xEnt loss between embeddings resulting from augmented and original input features. Evaluation is done using metrics like top5 retrieval accuracy, intra/inter-speaker distance and equal error rate.  

TODO: Results


## Data

The datasets used based on two widely used datasets: [Common Voice 17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) and [Voxceleb2](https://huggingface.co/datasets/ProgramComputer/voxceleb). 

### Common Voice 17

The dataset, created by the Mozilla foundation, consists of millions of recordings of short phrases spoken by many different speakers. The dataset is divided by language and into train, dev and test splits. In this project one dataset was assembled from the english train split and one from the dev split. In the metadata for this dataset there is also information about the up-votes and down-votes for each audio recording. This refers to users of the dataset and provides a rough estimation of the usefulness of each recording. 

The datasets were preprocessed into a form useful for the model using the following recipe:

1. The available recordings were filtered by a minimum number of upvotes and a maximum number of down votes
2. The remaining recordings were grouped by the speaker and filtered if there were too many or too few recordings for this speaker.
3. Each recording was resampled to 16 000 Hertz.
4. The [Whisper Feature Extractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py) was used to extract the input\_ids and attention\_maskof the recording. These two were saved into a [zarr](https://github.com/zarr-developers/zarr-python) archive along with some metadata like the speaker id and gender of the speaker.

| | Common Voice 17 en Train | Common Voice 17 en Dev |
| --- | --- | --- |
| Minimum up Votes | 2 | 2 |
| Maximum down Votes | 0 | 0 |
| Minimum recordings per speaker | 100 | 3 |
| Maximum recordings per speaker | 1000 | 3 |
| Distinct speakers | - | 2090 |
| Distinct recordings | - | 6270 |

### Voxceleb2



## Model


## Training 

## Results

## Conclusions


