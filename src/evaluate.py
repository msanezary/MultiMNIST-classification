import torch
import torch.nn.functional as F

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_top_left = 0
    correct_bottom_right = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_top_left = criterion(outputs[0], labels[:, 0])
            loss_bottom_right = criterion(outputs[1], labels[:, 1])
            loss = loss_top_left + loss_bottom_right
            total_loss += loss.item()

            _, predicted_top_left = torch.max(outputs[0], 1)
            _, predicted_bottom_right = torch.max(outputs[1], 1)

            correct_top_left += (predicted_top_left == labels[:, 0]).sum().item()
            correct_bottom_right += (predicted_bottom_right == labels[:, 1]).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy_top_left = 100 * correct_top_left / total
    accuracy_bottom_right = 100 * correct_bottom_right / total

    print(f'Evaluation - Loss: {avg_loss:.4f}, Accuracy Top-Left: {accuracy_top_left:.2f}%, Accuracy Bottom-Right: {accuracy_bottom_right:.2f}%')
    return avg_loss, accuracy_top_left, accuracy_bottom_right

def report_performance(model, test_loader, criterion, device='cpu'):
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()

    # Get evaluation metrics
    test_loss, accuracy_tl, accuracy_br = evaluate_model(model, test_loader, criterion, device)
    performance_summary = {
        'Test Loss': test_loss,
        'Accuracy Top-Left': accuracy_tl,
        'Accuracy Bottom-Right': accuracy_br
    }

    return performance_summary
