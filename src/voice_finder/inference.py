import argparse
import transformers
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import librosa
from sklearn.cluster import DBSCAN
from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=Path)
    parser.add_argument("--model", type=str, default="checkpoints/run_10/hf_model2")

    parser.add_argument("--threshold", type=float, default=0.8)

    return parser.parse_args()


def main():
    args = __parse_args()

    if not args.data.is_dir():
        raise ValueError("Data must be a path to a folder containing audio files.")

    # model = VoiceEmbedder.from_pretrained(args.model)

    # feature_extractor = VoiceEmbedderFeatureExtractor.from_pretrained(args.model)

    model = transformers.AutoModel.from_pretrained(args.model)
    feature_extractor = transformers.AutoProcessor.from_pretrained(args.model)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    model.to(device)

    all_audio = []
    all_sr = []
    all_filepaths = []
    for audio_path in args.data.iterdir():
        audio, sr = librosa.load(audio_path)
        all_audio.append(audio)
        all_sr.append(sr)
        all_filepaths.append(audio_path)

    features = feature_extractor(
        all_audio, all_sr, return_attention_mask=True, return_tensors="pt"
    )

    features = {k: v.to(device) for k, v in features.items()}

    with torch.no_grad():
        embeddings = model(**features)

    dists = torch.cdist(embeddings, embeddings).cpu().detach().numpy()

    dbscan = DBSCAN(eps=args.threshold, min_samples=1, metric="precomputed")
    clusters = dbscan.fit_predict(dists)

    cluster_order = np.argsort(clusters + (clusters == -1) * (np.max(clusters) + 1))
    reordered_matrix = dists[cluster_order][:, cluster_order]
    reordered_labels = [f"E{i + 1}" for i in cluster_order]

    cluster_dict = {}
    for cluster_id, filepath in zip(clusters, all_filepaths):
        if cluster_id not in cluster_dict:
            cluster_dict[int(cluster_id)] = []
        cluster_dict[int(cluster_id)].append(str(filepath))

    with open("clusters.json", "w") as f:
        json.dump(cluster_dict, f, indent=2)

    # Plot heatmap of reordered distance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        reordered_matrix,
        xticklabels=reordered_labels,
        yticklabels=reordered_labels,
        cmap="viridis",  # Low distances (same speaker) dark, high distances light
        square=True,
        cbar_kws={"label": "Distance"},
    )
    plt.title("Distance Matrix Heatmap (Ordered by DBSCAN Clusters, eps=0.8)")
    plt.xlabel("Embeddings")
    plt.ylabel("Embeddings")
    plt.tight_layout()
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()
