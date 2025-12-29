import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from models.gcn import GCN
from train import train
from evaluate import evaluate

def main():
    print("=== Project: IMBG2G | Phase 1: Baseline GCN on MUTAG ===")
    
    # Dataset
    dataset = TUDataset(root="IMBG2G/data/MUTAG/MUTAG", name="MUTAG")
    print(f"Total graphs in dataset: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        num_classes=dataset.num_classes
    ).to(device)
    print("Model initialized:", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, 51):
        print(f"\n--- Epoch {epoch} ---")
        loss = train(model, loader, optimizer, criterion, device)
        print(f"Training loss: {loss:.4f}")

        acc, f1 = evaluate(model, loader, device)
        print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

if __name__ == "__main__":
    main()
