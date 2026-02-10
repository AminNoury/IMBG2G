import torch
from itertools import combinations
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import settings as setting
from loss.imbalance_loss import ImbalanceLoss
from utils.utils import compute_class_weights

from g2g.gcn import GCN
from train import train
from evaluate import evaluate


def run_experiment(name, dataset, device):
    print(f"\n===== Running Experiment: {name} =====")

    dataset = dataset.shuffle()
    print(f"Total dataset size: {len(dataset)} | Num features: {dataset.num_node_features} | Num classes: {dataset.num_classes}")

    train_dataset = dataset
    val_dataset   = dataset
    test_dataset  = dataset

    if setting.TEST_MODE:
        train_dataset = train_dataset[:40]
        val_dataset   = val_dataset[:10]
        test_dataset  = test_dataset[:10]
        print("TEST MODE ACTIVE: Using small dataset subset")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32)
    test_loader  = DataLoader(test_dataset, batch_size=32)

    # ------------------------
    # Model
    # ------------------------
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        out_channels=dataset.num_classes
    ).to(device)
    print(model)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters())}")

    if setting.RUN_G2G:
        print("G2G Module ENABLED (placeholder)")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    print(f"Optimizer: Adam | LR: {optimizer.param_groups[0]['lr']} | Weight decay: 5e-4")

    # ------------------------
    # Loss Function
    # ------------------------
    if "imbalance" in name:
        class_weights = compute_class_weights(train_dataset).to(device)
        criterion = ImbalanceLoss(
            weight=class_weights,
            use_focal="focal" in name,
            gamma=setting.FOCAL_GAMMA
        )
        if "focal" in name:
            print(f"Using Imbalance Loss WITH Focal | Gamma: {setting.FOCAL_GAMMA}")
        else:
            print("Using Imbalance Loss WITHOUT Focal")
        print(f"Class weights: {class_weights}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("Using Standard CrossEntropy (Baseline)")

    # ------------------------
    # Train
    # ------------------------
    best_model_state, best_f1 = train(
        model, train_loader, optimizer, criterion, device,
        val_loader=val_loader, num_epochs=50
    )

    model.load_state_dict(best_model_state)

    # ------------------------
    # Evaluate
    # ------------------------
    acc, f1 = evaluate(model, test_loader, device)
    print(f"Test Result → Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1


def main():
    print("=== IMBG2G Ablation Study on MUTAG ===")
    dataset = TUDataset(root="IMBG2G/data/MUTAG/MUTAG", name="MUTAG")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # Baseline Experiments
    # ------------------------
    baseline_experiments = [
        "baseline",
        "baseline+imbalance",
        "baseline+imbalance+focal"
    ]

    results = {}

    # 1️⃣ Run baseline & imbalance experiments without any augmentation
    for exp_name in baseline_experiments:
        setting.AUG_NODE_DROP = False
        setting.AUG_EDGE_PERTURB = False
        setting.AUG_FEATURE_MASK = False
        results[exp_name] = run_experiment(exp_name, dataset, device)

    # ------------------------
    # Augmentation Experiments
    # ------------------------
    aug_ops = ["NodeDrop", "EdgePerturb", "FeatureMask"]
    aug_combos = []

    # Single, double, triple combinations
    for r in range(1, len(aug_ops)+1):
        aug_combos.extend(combinations(aug_ops, r))

    for combo in aug_combos:
        setting.AUG_NODE_DROP = "NodeDrop" in combo
        setting.AUG_EDGE_PERTURB = "EdgePerturb" in combo
        setting.AUG_FEATURE_MASK = "FeatureMask" in combo

        exp_name = f"Aug-{'+'.join(combo)}"
        results[exp_name] = run_experiment(exp_name, dataset, device)

    # ------------------------
    # Print final comparison
    # ------------------------
    print("\n===== FINAL COMPARISON =====")
    for k, v in results.items():
        print(f"{k:30s} → Acc: {v[0]:.4f}, F1: {v[1]:.4f}")


if __name__ == "__main__":
    main()
