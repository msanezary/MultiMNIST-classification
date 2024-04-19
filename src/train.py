import torch

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs[0], labels[:, 0]) + criterion(outputs[1], labels[:, 1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

        evaluate(model, test_loader, criterion, device)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs[0], labels[:, 0]) + criterion(outputs[1], labels[:, 1])
            total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(loader)}')
