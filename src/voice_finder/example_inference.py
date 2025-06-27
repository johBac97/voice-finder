import argparse
import umap
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
import seaborn as sns
import torch
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
from voice_finder import infer


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="johbac/voice-embedder-base")
    parser.add_argument("--number-samples", type=int, default=300)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("output"))

    return parser.parse_args()


def main():
    args = __parse_args()

    model = AutoModel.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    dataset = (
        load_dataset(
            "MLCommons/peoples_speech",
            "clean",
            split="validation",
            streaming=True,
        )
        .skip(1000)
        .shuffle(buffer_size=1000, seed=42)
    )

    batch_size = 100
    total_samples = 0
    batch_audio = []
    batch_sr = []
    batch_ids = []

    ids = []
    embeddings = []
    for d in dataset:
        batch_audio.append(d["audio"]["array"])
        batch_sr.append(d["audio"]["sampling_rate"])
        batch_ids.append(d["id"])
        total_samples += 1

        if len(batch_audio) >= batch_size or total_samples >= args.number_samples:
            features = processor(
                batch_audio, sampling_rate=batch_sr, return_tensors="pt"
            )

            batch_embeddings = infer(model, features)

            embeddings.append(batch_embeddings.cpu())
            ids.extend(batch_ids)

            print(
                f"Processed batch of {len(batch_ids)} samples, total samples: {total_samples}"
            )

            batch_audio = []
            batch_sr = []
            batch_ids = []

        if total_samples >= args.number_samples:
            break

    embeddings = torch.cat(embeddings, dim=0)

    dists = torch.cdist(embeddings, embeddings)

    dbscan = DBSCAN(eps=args.threshold, min_samples=1, metric="precomputed")
    clusters = dbscan.fit_predict(dists)

    cluster_order = np.argsort(clusters + (clusters == -1) * (np.max(clusters) + 1))
    reordered_matrix = dists[cluster_order][:, cluster_order]
    reordered_labels = [f"E{i + 1}" for i in cluster_order]

    cluster_dict = {}
    for cluster_id, filepath in zip(clusters, ids):
        if cluster_id not in cluster_dict:
            cluster_dict[int(cluster_id)] = []
        cluster_dict[int(cluster_id)].append(str(filepath))

    with open(args.output / "clusters.json", "w") as f:
        json.dump(cluster_dict, f, indent=2)
    print(f"Cluster assignments saved to '{args.output / 'clusters.json'}'")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        reordered_matrix,
        xticklabels=reordered_labels,
        yticklabels=reordered_labels,
        cmap="viridis",
        square=True,
        cbar_kws={"label": "Distance"},
    )
    plt.title(
        f"Distance Matrix Heatmap (Ordered by DBSCAN Clusters, eps={args.threshold})"
    )
    plt.xlabel("Embeddings")
    plt.ylabel("Embeddings")
    plt.tight_layout()
    plt.savefig(str(args.output / "plot.png"))

    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    if -1 in unique_clusters:
        colors[np.where(unique_clusters == -1)[0]] = [0, 0, 0, 1]
    for cluster_id, color in zip(unique_clusters, colors):
        mask = clusters == cluster_id
        label = "Noise" if cluster_id == -1 else f"Speaker {cluster_id + 1}"
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=label,
            s=60,
            alpha=0.6,
        )
    plt.title("UMAP Visualization of Speaker Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(str(args.output / "umap_plot.png"))
    print(f"UMAP plot saved as '{args.output / 'umap_plot.png'}'")


if __name__ == "__main__":
    main()
