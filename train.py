import torch
from evaluate import evaluate

def train(model, train_loader, optimizer, criterion, device, val_loader=None, num_epochs=50):
    """
    Train GCN model with optional validation.
    Returns best model state and best F1 score on val_loader (if given).
    """
    best_model_state = None
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average training loss: {avg_loss:.4f}")

        # Optional validation
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                acc, f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1} | Validation Accuracy: {acc:.4f}, F1: {f1:.4f}")

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict()

    # If no validation, return last model
    if best_model_state is None:
        best_model_state = model.state_dict()

    return best_model_state, best_f1
