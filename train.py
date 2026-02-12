# import torch
# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
# import torch.nn.functional as F
# from utils.augmentations import apply_augmentations
# from config import settings as setting

# def train(model, train_loader, optimizer, criterion, device, val_loader=None, num_epochs=50):
#     """
#     Train GCN model with full logging + augmentation awareness.
#     Logs:
#     - Batch loss, sample logits/labels/confidence
#     - Epoch avg loss
#     - Validation Accuracy/F1 and Confusion Matrix
#     - Augmentation effect (NodeDrop/EdgePerturb/FeatureMask)
#     Returns best model state (based on F1) and best F1 score.
#     """

#     best_model_state = None
#     best_f1 = 0.0

#     print("\n=== START TRAINING ===")
#     print(f"Device: {device}")
#     print(f"Number of training batches: {len(train_loader)}")

#     print("\n=== AUGMENTATION SETTINGS ===")
#     print(f"Node Drop: {setting.AUG_NODE_DROP} | Ratio: {setting.NODE_DROP_RATIO}")
#     print(f"Edge Perturb: {setting.AUG_EDGE_PERTURB} | Ratio: {setting.EDGE_PERTURB_RATIO}")
#     print(f"Feature Mask: {setting.AUG_FEATURE_MASK} | Ratio: {setting.FEATURE_MASK_RATIO}")
#     print("=================================================\n")

#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         total_loss = 0

#         print(f"\n--- Epoch {epoch}/{num_epochs} ---")
#         print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

#         for batch_idx, data in enumerate(train_loader, 1):
#             data = data.to(device)

#             # -------- APPLY AUGMENTATION ONLY IN TRAIN --------
#             if setting.RUN_AUGMENTATION:
#                 data = apply_augmentations(data, setting)
#                 if batch_idx == 1:
#                     print(f"Augmentation applied: NodeDrop={setting.AUG_NODE_DROP}, EdgePerturb={setting.AUG_EDGE_PERTURB}, FeatureMask={setting.AUG_FEATURE_MASK}")

#             optimizer.zero_grad()
#             out = model(data)
#             loss = criterion(out, data.y)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#             # -------- BATCH LOGS --------
#             if batch_idx % 5 == 0 or batch_idx == len(train_loader):
#                 logits_np = out[:3].detach().cpu().numpy()
#                 labels_np = data.y[:3].detach().cpu().numpy()
#                 probs = F.softmax(out[:3], dim=1).detach().cpu().numpy()
#                 print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
#                 print(f"Sample logits (first 3):\n{logits_np}")
#                 print(f"Sample labels (first 3): {labels_np}")
#                 print(f"Sample confidence (first 3):\n{probs}")

#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch} | Average training loss: {avg_loss:.4f}")

#         # ---------------- VALIDATION ----------------
#         if val_loader is not None:
#             model.eval()
#             all_preds, all_labels = [], []

#             with torch.no_grad():
#                 for val_data in val_loader:
#                     val_data = val_data.to(device)
#                     val_out = model(val_data)
#                     preds = val_out.argmax(dim=1)
#                     all_preds.append(preds.cpu())
#                     all_labels.append(val_data.y.cpu())

#             all_preds = torch.cat(all_preds)
#             all_labels = torch.cat(all_labels)

#             acc = accuracy_score(all_labels, all_preds)
#             f1 = f1_score(all_labels, all_preds, average='macro')
#             cm = confusion_matrix(all_labels, all_preds)

#             print(f"Epoch {epoch} | Validation Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
#             print(f"Confusion Matrix:\n{cm}")

#             # Save best model
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_model_state = model.state_dict()
#                 print(f"New best model saved with F1: {best_f1:.4f} at epoch {epoch}")

#     # ---------------- FINALIZE ----------------
#     if val_loader is None:
#         print("\n⚠️ No validation loader provided. Returning last model state.")
#         best_model_state = model.state_dict()

#     print(f"\n=== TRAINING FINISHED | Best F1: {best_f1:.4f} ===")
#     return best_model_state, best_f1
import torch
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import torch.nn.functional as F
from utils.augmentations import apply_augmentations
from config import settings as setting
from g2g.build_graph_of_graphs import build_g2g_edges
from g2g.g2g_gnn import G2GNN
from g2g.graph_encoder import GraphEncoder

def train(model, train_loader, optimizer, criterion, device, val_loader=None, num_epochs=50):
    """
    Full-featured training loop for GCN + optional Augmentation + optional Graph-to-Graph (G2G).
    
    Logs:
    - Batch loss, sample logits/labels/confidence
    - Epoch average loss
    - Validation Accuracy, F1, Confusion Matrix
    - Augmentation effect (NodeDrop/EdgePerturb/FeatureMask)
    - G2G info (number of graph-to-graph edges, loss if used)
    
    Returns:
        best_model_state (dict): model state dict for best validation F1
        best_f1 (float): best validation F1 achieved
    """

    best_model_state = None
    best_f1 = 0.0

    print("\n=== START TRAINING ===")
    print(f"Device: {device}")
    print(f"Number of training batches: {len(train_loader)}")

    # ---------------- AUGMENTATION SETTINGS ----------------
    print("\n=== AUGMENTATION SETTINGS ===")
    print(f"Node Drop: {setting.AUG_NODE_DROP} | Ratio: {setting.NODE_DROP_RATIO}")
    print(f"Edge Perturb: {setting.AUG_EDGE_PERTURB} | Ratio: {setting.EDGE_PERTURB_RATIO}")
    print(f"Feature Mask: {setting.AUG_FEATURE_MASK} | Ratio: {setting.FEATURE_MASK_RATIO}")
    print("=================================================\n")

    # ---------------- G2G SETTINGS ----------------
    if setting.RUN_G2G:
        print("\n=== G2G SETTINGS ===")
        print(f"Use KNN Graph: {setting.G2G_USE_KNN_GRAPH}")
        print(f"Use Message Passing: {setting.G2G_USE_MESSAGE_PASSING}")
        print(f"K neighbors: {setting.G2G_K}, Hidden Dim: {setting.G2G_HIDDEN_DIM}")
        print("=================================================\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        for batch_idx, data in enumerate(train_loader, 1):
            data = data.to(device)

            # ---------------- APPLY AUGMENTATION ----------------
            if setting.RUN_AUGMENTATION:
                data = apply_augmentations(data, setting)
                if batch_idx == 1:
                    print(f"Augmentation applied: NodeDrop={setting.AUG_NODE_DROP}, EdgePerturb={setting.AUG_EDGE_PERTURB}, FeatureMask={setting.AUG_FEATURE_MASK}")

            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)

            # ---------------- OPTIONAL G2G ----------------
            if setting.RUN_G2G and setting.G2G_USE_MESSAGE_PASSING:
                # Encode batch graphs into embeddings
                graph_encoder = GraphEncoder(
                    in_channels=data.num_node_features,
                    hidden_channels=setting.G2G_HIDDEN_DIM
                ).to(device)
                graph_emb = graph_encoder(data.x, data.edge_index, data.batch)

                # Build graph-of-graphs edges
                if setting.G2G_USE_KNN_GRAPH:
                    g2g_edge_index = build_g2g_edges(graph_emb, k=setting.G2G_K)
                    print(f"G2G edges built: {g2g_edge_index.size(1)} edges")

                    # G2G message passing
                    g2g_model = G2GNN(
                        in_channels=graph_emb.size(1),
                        hidden_channels=setting.G2G_HIDDEN_DIM,
                        num_classes=out.size(1)
                    ).to(device)
                    g2g_out = g2g_model(graph_emb, g2g_edge_index)
                    # Combine original loss + G2G (simple sum)
                    loss += criterion(g2g_out, data.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # ---------------- BATCH LOGS ----------------
            if batch_idx % 5 == 0 or batch_idx == len(train_loader):
                logits_np = out[:3].detach().cpu().numpy()
                labels_np = data.y[:3].detach().cpu().numpy()
                probs = F.softmax(out[:3], dim=1).detach().cpu().numpy()
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                print(f"Sample logits (first 3):\n{logits_np}")
                print(f"Sample labels (first 3): {labels_np}")
                print(f"Sample confidence (first 3):\n{probs}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Average training loss: {avg_loss:.4f}")

        # ---------------- VALIDATION ----------------
        if val_loader is not None:
            model.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for val_data in val_loader:
                    val_data = val_data.to(device)
                    val_out = model(val_data)
                    preds = val_out.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(val_data.y.cpu())

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            cm = confusion_matrix(all_labels, all_preds)

            print(f"Epoch {epoch} | Validation Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")

            # ---------------- SAVE BEST MODEL ----------------
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()
                print(f"New best model saved with F1: {best_f1:.4f} at epoch {epoch}")

    # ---------------- FINALIZE ----------------
    if val_loader is None:
        print("\n⚠️ No validation loader provided. Returning last model state.")
        best_model_state = model.state_dict()

    print(f"\n=== TRAINING FINISHED | Best F1: {best_f1:.4f} ===")
    return best_model_state, best_f1
