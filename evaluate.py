import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)

            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")

    print(f"Evaluation -> Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
    return acc, f1
