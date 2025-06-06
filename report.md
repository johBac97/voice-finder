# Voice Speaker Identification using Pretrained Whisper encoder and Contrastive Learning

The goal of this project is to develop a model capable of producing speaker embeddings — compact, fixed-length vector representations that capture the unique vocal characteristics of individual speakers. Such embeddings are useful for a range of downstream tasks, including speaker verification, clustering, and voice similarity retrieval. The core idea is that utterances from the same speaker should yield similar embeddings, while utterances from different speakers should be well-separated in the embedding space.

To train such model we use a robust pretrained audio encoder model and finetune it to capture speaker identity information using a contrastive learning approach. For the encoder model we opted to use the *Whisper Base* encoder. Whisper is a robust widely adopted transcription model developed by OpenAI. It is robust accross accents and background noise and is therefore a suitable base model.

The contrastive learning approach leverages a hard-triplet mining technique to provide the most useful training examples for the model. Additionally, a regularizing self supervised learning objective is introduced using NT-xEnt loss between embeddings resulting from augmented and original input features. Evaluation is done using metrics like top5 retrieval accuracy, intra/inter-speaker distance and equal error rate.  

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

The *Normalized Temperature-scaled Cross Entropy Loss* (NT-Xent), introduced in [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709), is a contrastive loss that encourages embeddings from augmented versions of the same input to be close in the embedding space, while pushing apart embeddings from different samples.

Given a batch of $\( N \)$ input samples, two augmented views are created for each sample, resulting in $\( 2N \)$ embeddings: $\( \{ z_1^i, z_2^i \}_{i=1}^N \)$. These are concatenated into a single batch of size $\( 2N \)$. For each anchor embedding $\( z_i \)$, there is a corresponding positive embedding $\( z_j \)$, while the remaining $\( 2N - 2 \)$ embeddings are treated as negatives.

The cosine similarity between two embeddings is computed and scaled by a temperature parameter $\( \tau \)$:

$$
\text{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\tau}
$$

To normalize the similarity scores for contrastive learning, we define the softmax denominator (excluding the anchor itself) as:

$$
\mathcal{S}_i = \sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp(\text{sim}(z_i, z_k))
$$

The NT-Xent loss for a single positive pair \( (z_i, z_j) \) is then given by:

$$
\mathcal{L}_i = -\log \frac{\exp(\text{sim}(z_i, z_j))}{\mathcal{S}_i}
$$

Aggregating over all \( 2N \) embeddings, the total loss becomes:

$$
\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{i=1}^{2N} \mathcal{L}_i
$$

In this experiment, we set the temperature to $\( \tau = 0.5 \)$. This objective acts as a self-supervised regularizer, enforcing invariance under augmentation and improving the generalization of the learned embedding space.

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


### Dataset Implementation

To support speaker-aware training, a custom PyTorch-style dataset was implemented. Each dataset sample corresponds to a **unique speaker**, and when sampled, it returns a **fixed number of utterances** (in this case, four) from that speaker. The utterances are selected randomly from the set of available recordings for that speaker.

The dataset is backed by a [Zarr](https://zarr.readthedocs.io/) archive containing precomputed Whisper features, attention masks, and metadata such as speaker ID and gender. At runtime, two types of data augmentations are optionally applied:

- **Triplet augmentations**, applied to the original features to improve robustness of the supervised triplet loss.
- **NT-Xent augmentations**, applied separately to generate perturbed versions of the features for contrastive learning.

This design enables the training loop to compute both the NT-Xent loss and hard-mining triplet loss from a unified data structure while maintaining speaker-level grouping and efficient access.

### Data Augmentations


To improve generalization and robustness of the embedding model, several data augmentations were implemented and applied during training. Each augmentation operates on precomputed log-Mel spectrogram features and is randomly sampled during training:

- **Gaussian Noise**: Adds i.i.d. Gaussian noise to all feature values. This simulates sensor or environment noise and encourages stability under small perturbations.

- **Time Masking**: Randomly selects a temporal segment within the input and replaces it with noise. This simulates dropouts or occlusions in time, similar to SpecAugment.

- **Frequency Masking**: Randomly selects a frequency band and replaces it with noise. This forces the model to rely on broader frequency context and is also inspired by SpecAugment.

- **Low-Frequency Noise**: Adds structured noise to the lowest frequency bands only, emulating interference such as background hum or reverberation in voice channels.

Each augmentation returns a modified version of the features and leaves the attention mask untouched. They can be applied independently or in combination, both for contrastive (NT-Xent) and supervised (triplet) learning objectives.


## Results

### Training Runs

Below are the results of the training runs. The Augmentations have been abbreviated as follows:

* GN : Guassian Noise
* TM : Time Mask
* FM : Frequence Mask
* LFN : Low Frequency Noise

| Run Name | Effective Batch Size | Self-Supervised Loss Weight | Gendered | # NxEnt Augmentations | # Triplet Augmentations | NxEnt Augmentations     | Triplet Augmentations |
|----------|----------------------|-----------------------------|----------|------------------------|--------------------------|--------------------------|------------------------|
| Run 1    | 32 * 4               | 0.0                         | False    | 0                      | 0                        | []                       | []                     |
| Run 2    | 32 * 4               | 0.0                         | True     | 0                      | 0                        | []                       | []                     |
| Run 3    | 32 * 4               | 0.5                         | False    | 3                      | 0                        | [GN, TM, FM, LFN]        | []                     |
| Run 4    | 32 * 4               | 0.5                         | True     | 3                      | 0                        | [GN, TM, FM, LFN]        | []                     |
| Run 5    | 32 * 4               | 0.5                         | False    | 1                      | 0                        | [GN, TM, FM, LFN]        | []                     |
| Run 6    | 32 * 4               | 0.1                         | False    | 1                      | 0                        | [GN, TM, FM, LFN]        | []                     |
| Run 7    | 32 * 4               | 0.025                       | False    | 1                      | 0                        | [GN, TM, FM, LFN]        | []                     |
| Run 8    | 32 * 4               | 0.025                       | False    | 3                      | 0                        | [GN, TM, FM, LFN]        | []                     |
| Run 9    | 32 * 4               | 0.0                         | False    | 0                      | 1                        | []                       | []                     |
| Run 10   | 32 * 4               | 0.025                       | False    | 1                      | 0                        | [GN]                     | []                     |

### Metrics

Here is the eval results of these training runs for the __Voxceleb2 en dev__ dataset.

| run    | Average Same L2 Distance | Average Different L2 Distance | Top 1 Accuracy | Top 5 Accuracy |  Equal Error Rate  |
|--------|-------------------|-------------------|----------------|----------------|-------|
| run 1  | 0.7970            | 1.1481            | 0.1428         | 0.2316         | 0.1788 |
| run 2  | 0.8310            | 1.1587            | 0.1066         | 0.1809         | 0.2117 |
| run 3  | 0.9205            | 1.2491            | 0.1314         | 0.2157         | 0.1941 |
| run 4  | 0.9163            | 1.2148            | 0.0985         | 0.1762         | 0.2310 |
| run 5  | 0.9072            | 1.2471            | 0.1279         | 0.2135         | 0.1918 |
| run 6  | 0.8982            | 1.2394            | 0.1239         | 0.2078         | 0.1821 |
| run 7  | 0.8338            | 1.1669            | 0.1331         | 0.2255         | 0.1817 |
| run 8  | 0.8375            | 1.1685            | 0.1331         | 0.2220         | 0.1857 |
| run 9  | 0.7849            | 1.1642            | 0.1199         | 0.2083         | 0.1801 |
| run 10 | 0.8152            | 1.1514            | 0.1421         | 0.2287         | 0.1820 |


And here is the results for the runs for the __Common Voice 17 en dev__  dataset.

| run       | Average Same L2 Distance | Average Different L2 Distance | Top 1 Accuracy | Top 5 Accuracy |  Equal Error Rate  |
|-----------|-------------------|-------------------|----------------|----------------|-------|
| run 1  | 0.5375            | 1.3571            | 0.9448         | 0.9833         | 0.0110 |
| run 2  | 0.4943            | 1.3515            | 0.8968         | 0.9617         | 0.0176 |
| run 3  | 0.5503            | 1.3817            | 0.9266         | 0.9762         | 0.0120 |
| run 4  | 0.5478            | 1.3798            | 0.8882         | 0.9563         | 0.0187 |
| run 5  | 0.5682            | 1.3862            | 0.9322         | 0.9769         | 0.0120 |
| run 6  | 0.5732            | 1.3744            | 0.9187         | 0.9735         | 0.0128 |
| run 7  | 0.5647            | 1.3682            | 0.9274         | 0.9732         | 0.0120 |
| run 8  | 0.5543            | 1.3676            | 0.9321         | 0.9769         | 0.0130 |
| run 9  | 0.4737            | 1.3266            | 0.9222         | 0.9718         | 0.0142 |
| run 10 | 0.5456            | 1.3617            | 0.9413         | 0.9817         | 0.0105 |


### Plots


<table>
  <tr>
    <td><img src="resources/plots/run_1/common_voice_17_en_dev/distance_histograms.png" width="200px"></td>
    <td><img src="resources/plots/run_1/voxceleb2_en_dev/distance_histograms.png" width="200px"></td>
  </tr>
  <tr>
    <td align="center">Common Voice 17 en Dev</td>
    <td align="center">Voxceleb2 en Dev</td>
  </tr>
</table>

__Figure 1__ : Intera/inter embedding distances for the model trained in __run 1__.


<table>
  <tr>
    <td><img src="resources/plots/run_4/common_voice_17_en_dev/distance_histograms.png" width="200px"></td>
    <td><img src="resources/plots/run_4/voxceleb2_en_dev/distance_histograms.png" width="200px"></td>
  </tr>
  <tr>
    <td align="center">Common Voice 17 en Dev</td>
    <td align="center">Voxceleb2 en Dev</td>
  </tr>
</table>

__Figure 2__ : Intera/inter embedding distances for the model trained in __run 4__.

<table>
  <tr>
    <td><img src="resources/plots/run_10/common_voice_17_en_dev/distance_histograms.png" width="200px"></td>
    <td><img src="resources/plots/run_10/voxceleb2_en_dev/distance_histograms.png" width="200px"></td>
  </tr>
  <tr>
    <td align="center">Common Voice 17 en Dev</td>
    <td align="center">Voxceleb2 en Dev</td>
  </tr>
</table>

__Figure 3__ : Intera/inter embedding distances for the model trained in __run 10__.

## Analysis

This project proved that it is possible to train an efficient voice speaker embedding model using a contrastive learning approach. The very high top 1 and top 5 retrieval accuracy on the test set of the Common Voice 17 en dataset proves that the trained model learns useful features to effectivly distinguish speakers. However, the impressive performance does not transfer well to the other Voxceleb2 en test set. This is likely due to the very differnt nature of the recordings in the two datasets. Common Voice 17 contains studio recordings with quiet surroundings and clear pronunciations. On the other hand, Voxceleb2 en contains audio tracks scraped from public videos on platforms such as youtube and often depicts real world celebrity interviews. Such interviews or often conducted on the street or in crowded areas with lots of background noise and other background speakers. The trained model struggles to capture speaker information as effectivly in this setting as in the studio setting. Still, a top 5 retrieval accuracy of ~20 % shows that the model captures some speaker information correctly and far outperforms a random baseline wich would result in a top-5 accuracy of less than 0.1 %. 

Intrestingly, the addition of the self supervised loss does not increase retrieval accuracy on the test sets, if anything it decreases it. The idea with the self supervised loss is to allow the model to train on generating more robust embeddings, withstanding things such as noise and loss of signal. However, since the training objective is joint it also removes some focus from the triplet loss, which focuses on distinguishing speakers. To mitigate this the self supervised loss weight was lowered substantially in later experiments, but with similar results. It was thought that the self supervised loss would help especially for the very noisy test set Voxceleb2 en dev. The fact that it didn't leads to the suspicion that either 1. the noise parameters were not properly calibrated resulting in too hard of a subtask which just confused the model. 2. The augmentations used does not reflect realistic conditions and therefore doesn't improve model robustness. Given that the sub loss for the self supervised part of the training objective decreased steadily during training until reacing a steady state, i.e, meaning that the model was able to learn to complete the task, it seems that the second option is more likely. The augmentations used, Time Mask, Frequency Mask, Low Freuency Noise, Guassian Noise, are all standard augmentations but it seems that they do not properly reflect the kind of noise present in the test recordings. In the Voxceleb2 en dev dataset most noise comes from surrounding ambient city sounds like cars, car horns, background voices various harsh clanking sounds. In the next version of this work these types of noises should be modeled using augmentations.

## Future Works

1. Model real world noise using more advanced augmentations 
2. Extend training dataset with samples recorded in noisy environments.




