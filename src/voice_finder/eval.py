from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_curve

import torch
from tqdm import tqdm
import argparse
from voice_finder import VoiceEmbedder, VoiceEmbedderDataset, collate_fn


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--results", type=Path)

    return parser.parse_args()


def run_inference(model, dataloader, device):
    model = model.to(device)
    model.eval()
    all_embeddings = []
    client_indices = []
    client_ids = []
    paths = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Inference"):
        with torch.no_grad():
            input_features = batch["features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embeddings = model(
                input_features=input_features, attention_mask=attention_mask
            )

            all_embeddings.append(embeddings.to("cpu"))
            client_indices.append(batch["client_index"].to("cpu"))
            client_ids.extend(batch["client_id"])
            paths.extend(batch["path"])

    embeddings = torch.cat(all_embeddings)
    indices = torch.cat(client_indices)

    return embeddings, indices, client_ids, paths


def evaluate_embeddings(embeddings, indices, top_k=5, device=None):
    if device is None:
        device = embeddings.device

    # Compute pairwise L2 distances
    dists = torch.cdist(embeddings, embeddings, p=2)  # Shape: [N, N]

    # Create masks
    same_client = indices.unsqueeze(0) == indices.unsqueeze(0).T
    eye_mask = torch.eye(len(indices), device=device, dtype=torch.bool)
    triu_mask = torch.triu(torch.ones_like(same_client), diagonal=1).bool()

    # Average distances
    avg_same_dist = dists[same_client & ~eye_mask].mean()
    avg_diff_dist = dists[~same_client & ~eye_mask].mean()

    # Top-k and Top-1 retrieval accuracy
    sim = -dists.clone()
    sim[eye_mask] = -float("inf")  
    topk_indices = torch.topk(sim, k=top_k, dim=1).indices
    topk_match = (indices[topk_indices] == indices.unsqueeze(1)).any(dim=1)
    top1_match = indices[topk_indices[:, 0]] == indices

    # Equal Error Rate (EER)
    true_labels = same_client[triu_mask].cpu().numpy().astype(int)
    dist_scores = dists[triu_mask].cpu().numpy()
    fpr, tpr, _ = roc_curve(true_labels, -dist_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    embedding_norms = embeddings.norm(p=2, dim=1)
    mean_norm = embedding_norms.mean()
    std_norm = embedding_norms.std()

    print(f"Average same-speaker distance: {avg_same_dist.item():.4f}")
    print(f"Average different-speaker distance: {avg_diff_dist.item():.4f}")
    print(f"Top-1 retrieval accuracy: {top1_match.float().mean().item():.4f}")
    print(f"Top-{top_k} retrieval accuracy: {topk_match.float().mean().item():.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f}")
    print(f"Embedding Norm mean (std): {mean_norm.item():.4f} ({std_norm.item():.4f})")

    return {
        "avg_same_distance": avg_same_dist.item(),
        "avg_diff_distance": avg_diff_dist.item(),
        "top_1_accuracy": top1_match.float().mean().item(),
        f"top_{top_k}_accuracy": topk_match.float().mean().item(),
        "eer": float(eer),
        "embedding_norm_mean": mean_norm.item(),
        "embedding_norm_std": std_norm.item(),
    }


def main():
    args = __parse_args()

    model = VoiceEmbedder.from_pretrained(args.model)

    dataset = VoiceEmbedderDataset(
        args.dataset,
        sequence_length=model.config.sequence_length,
        all_samples_per_client=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn
    )

    embeddings, indices, client_ids, paths = run_inference(
        model, dataloader, "cuda" if torch.cuda.is_available() else "cpu"
    )

    results = evaluate_embeddings(embeddings, indices, top_k=5, device=None)

    if args.results:
        with args.results.open("w") as io:
            json.dump(
                {**results, "model": args.model, "dataset": args.dataset},
                io,
                indent=4,
                default=str,
            )


if __name__ == "__main__":
    main()
