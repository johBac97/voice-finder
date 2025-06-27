import torch


def infer(
    model,
    features,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device).eval()

    embeddings = []
    for start_index in range(0, features["input_features"].shape[0], batch_size):
        batch = {
            k: v[start_index : start_index + batch_size, :].to(device)
            for k, v in features.items()
        }

        with torch.no_grad():
            embeddings.extend(model(**batch))
    embeddings = torch.stack(embeddings).detach().cpu()

    return embeddings
