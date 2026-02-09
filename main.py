import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from g2g.gcn import GCN
from train import train
from evaluate import evaluate

def main():
    print("=== Project: IMBG2G | Phase 1: Baseline GCN on MUTAG ===")
    
    # ------------------------
    # Dataset
    # ------------------------
    dataset = TUDataset(root="IMBG2G/data/MUTAG/MUTAG", name="MUTAG")
    dataset = dataset.shuffle()

    # Split
    train_dataset = dataset[:150]
    val_dataset   = dataset[150:170]
    test_dataset  = dataset[170:]

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)
    test_loader  = DataLoader(test_dataset, batch_size=32)

    print(f"Total graphs: {len(dataset)}")
    print(f"Node features: {dataset.num_node_features}, Classes: {dataset.num_classes}")

    # ------------------------
    # Device
    # ------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # Model
    # ------------------------
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes  # اصلاح نام پارامتر
    ).to(device)
    print("Model initialized")

    # ------------------------
    # Optimizer & Loss
    # ------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005,         # کمی کاهش LR برای دیتاست کوچک
        weight_decay=5e-4 # جلوگیری از overfitting
    )
    criterion = torch.nn.CrossEntropyLoss()

    # ------------------------
    # Train model
    # ------------------------
    best_model_state, best_f1 = train(
        model, train_loader, optimizer, criterion, device,
        val_loader=val_loader, num_epochs=50
    )
    model.load_state_dict(best_model_state)
    print("Training completed, best model loaded")

    # ------------------------
    # Evaluate on test set
    # ------------------------
    acc, f1 = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {acc:.4f} | F1-score: {f1:.4f}")


if __name__ == "__main__":
    main()
