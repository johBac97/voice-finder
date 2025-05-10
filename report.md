# Voice Speaker Identification using Pretrained Whisper encoder and Contrastive Learning

The goal of this project is to develop a model capable of producing speaker embeddings — compact, fixed-length vector representations that capture the unique vocal characteristics of individual speakers. Such embeddings are useful for a range of downstream tasks, including speaker verification, clustering, and voice similarity retrieval. The core idea is that utterances from the same speaker should yield similar embeddings, while utterances from different speakers should be well-separated in the embedding space.

To train such model we use a robust pretrained audio encoder model and finetune it to capture speaker identity information using a contrastive learning approach. For the encoder model we opted to use the *Whisper Base* encoder. Whisper is a robust widely adopted transcription model developed by OpenAI. It is robust accross accents and background noise and is therefore a suitable base model.

The contrastive learning approach leverages a hard-triplet mining technique to provide the most useful training examples for the model. Additionally, a regularizing self supervised learning objective is introduced using NT-xEnt loss between embeddings resulting from augmented and original input features. Evaluation is done using metrics like top5 retrieval accuracy, intra/inter-speaker distance and equal error rate.  

TODO: Results


## Data

The datasets used based on two widely used datasets: [Common Voice 17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) and [Voxceleb2](https://huggingface.co/datasets/ProgramComputer/voxceleb). 

### Common Voice 17

The dataset, created by the Mozilla foundation, consists of millions of recordings of short phrases spoken by many different speakers. The samples are recorded in what seems to be a studio environment and are generally of high audio quality. The dataset is divided by language and into train, dev and test splits. In this project one dataset was assembled from the english train split and one from the dev split. In the metadata for this dataset there is also information about the up-votes and down-votes for each audio recording. This refers to users of the dataset and provides a rough estimation of the usefulness of each recording. 

The datasets were preprocessed into a form useful for the model using the following recipe:

1. The available recordings were filtered by a minimum number of upvotes and a maximum number of down votes
2. The remaining recordings were grouped by the speaker and filtered if there were too many or too few recordings for this speaker.
3. Each recording was resampled to 16 000 Hertz.
4. The [Whisper Feature Extractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py) was used to extract the input\_ids and attention\_maskof the recording. These two were saved into a [zarr](https://github.com/zarr-developers/zarr-python) archive along with some metadata like the speaker id and gender of the speaker.

### Voxceleb2

This dataset, created by Oxford University, consists of audio and video recordings of celebrities extracted from youtube.com. The speakers come from a different nations speaking different languages and dialects. The recordings are also vary widly in quality and background noise. It consists of a dev and a test split. In this project only the dev split was used. 

Since the training data in the Common Voice dataset only contains english speech, the Voxceleb2 dev dataset had to be filtered to only contain english samples. To accomplish this a pretrained language recognition model was used to determine the language of each sample. The model selected for this task was the `speechbrain/lang-id-voxlingua107-ecapa` model. The result of this subtask was uploaded as a metadata dataset to [huggingface](https://huggingface.co/datasets/johbac/voxceleb-language-metadata). After this filtering 537 134 audio recordings remained from 4575 unique speakers.

To preprocess this filtered version of the Voxceleb2 dev split dataset into a format consumable by the model the following steps were performed:

1. All samples from speakers with a total number of recordings less than three were removed
2. For each remaining speaker 3 samples were sampled and stored in the dataset. All other recordings from each speaker were discarded.
3. Each recording was resampled to 16 000 Hertz.
4. The [Whisper Feature Extractor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/feature_extraction_whisper.py) was used to extract the input\_ids and attention\_maskof the recording. These two were saved into a [zarr](https://github.com/zarr-developers/zarr-python) archive.

This sampling of three recordings per speaker was made to provide a uniform dataset to measure retrieval accuracy.

### Datasets Statistics

Below is a table with statistics about the three datasets used in this project.

| | Common Voice 17 en Train | Common Voice 17 en Dev | Voxceleb2 dev|
| --- | --- | --- | --- |
| Minimum up Votes | 2 | 2 | - |
| Maximum down Votes | 0 | 0 | - |
| Minimum recordings per speaker | 100 | 3 | 3 |
| Maximum recordings per speaker | 1000 | 3 | 3 |
| Distinct speakers | 1257 | 2090 | 4252 |
| Distinct recordings | 305706 | 6270 | 12756 |


## Model

The model used consists of two parts: first the encoder of the `openai/Whisper-base` model on [huggingface](https://huggingface.co/openai/whisper-base). Secondly, a MLP projector head consisting of two linear layers with a ReLU action in between. The project head transforms the last hidden state of the encoder model of dimension 512 to an embedding with dimenision 256.

## Training 

The training procedure was heavily inspired by the paper [Whisper Speaker Identification: Leveraging Pre-Trained Multilinguial Transformers for Robust Speaker Embeddings](https://arxiv.org/html/2503.10446) which used a combined loss function along with random augmentations to train speaker embeddings.

<!-- The training was conducted using a GeForce RTX 3090 with 24 GB of VRAM. -->

### Loss Function

The training objective combines two complementary loss functions:

1. **A self-supervised contrastive loss** (`NT-Xent`) applied between augmented views of the same audio sample.
2. **A supervised hard-mining triplet loss**, which uses speaker labels to construct difficult positive and negative examples.

#### NT-Xent Loss

The *Normalized Temperature-scaled Cross Entropy Loss* (NT-Xent), introduced in [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709), is a contrastive loss that encourages embeddings from augmented versions of the same input to be close, while pushing apart embeddings from other samples.

Given a batch of $\( N \)$ input samples, we create two views of each sample via data augmentation, resulting in $\( 2N \)$ embeddings: $\( { z_1^i, z_2^i }_{i=1}^N \)$. Let $\( z_i \)$ be the anchor and $\( z_j \)$ be the positive pair, and all others in the batch are treated as negatives. The similarity is computed using cosine similarity scaled by a temperature parameter $\( \tau \)$:

$$
\text{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\tau}
$$

The loss for each anchor $\( z_i \)$ is computed using cross-entropy over the similarity scores with respect to its positive pair:

$$
\mathcal{L}_{\text{NT-Xent}} = -\log \frac{\exp(\text{sim}(z_i, z_j))}{\sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp(\text{sim}(z_i, z_k))}
$$

In this experiment, we use cosine similarity and $\( \tau = 0.5 \)$.

This objective helps regularize the embedding space by enforcing consistency under small perturbations and improves generalization.

#### Hard-Mining Triplet Loss

The second part of the training objective is a supervised *triplet loss* that uses known speaker identities to construct hard triplets within a batch.

Each training sample is treated as an *anchor*. The *positive* is the most dissimilar embedding (i.e., largest distance) from the same speaker, and the *negative* is the most similar embedding (i.e., smallest distance) from a different speaker.

For a given anchor $\( a \)$, positive $\( p \)$, and negative $\( n \)$, the loss is defined as:

$$
\mathcal{L}_{\text{triplet}} = \max(0, \, d(a, p) - d(a, n) + \alpha)
$$

where $\( d(a, b) \)$ is the Euclidean distance between embeddings $\( a \)$ and $\( b \)$, and $\( \alpha \)$ is the margin (set to 1.0 in this experiment).

This *online hard-mining* strategy forces the model to learn to discriminate between closely spaced speakers and adapt to the hardest cases within each mini-batch.

#### Combined Loss

The total training loss $\( \mathcal{L} \)$ is a weighted combination of the two:

$$
\mathcal{L} = \lambda \cdot \mathcal{L}_{\text{NT-Xent}} + \mathcal{L}_{\text{triplet}}
$$

where $\( \lambda \)$ is a hyperparameter controlling the contribution of the NT-Xent loss (tuned during experiments).

This joint objective aims to simultaneously benefit from both *self-supervised regularization* and *supervised discrimination*, leading to embeddings that are both stable and speaker-discriminative.



## Results

## Conclusions


