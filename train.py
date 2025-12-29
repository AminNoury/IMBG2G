import torch

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if i % 5 == 0:
            print(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Average training loss this epoch: {avg_loss:.4f}")
    return avg_loss
