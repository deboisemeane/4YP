import torch


def calculate_ce_loss(model, cross_entropy_criterion, dataloader, device):  # Calculates the average loss over the whole dataset

    # This function averages loss correctly if criterion.reduce='mean' i.e. criterion is already averaging over each batch.

    criterion = cross_entropy_criterion
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for i, batch in enumerate(dataloader):
            x = batch["features"].to(device)  # Move features to device
            labels = batch["label"].to(device)  # Move labels to device

            y = model(x)
            loss = criterion(y, labels)

            total_loss += loss.item()

            _, predicted_idx = y.max(1)  # Finds the index of the predicted class. (0:N3, 1:N1/N2, 2:REM, 3:W)

            total_correct += (predicted_idx == labels).sum().item() / x.shape[0]  # Divide by batch size so we can average over batches

    average_loss = total_loss / len(dataloader)  # Divides by the number of batches, since CrossEntropyLoss.reduce='mean'
    # by default, so values are already averaged within each batch
    accuracy = total_correct / len(dataloader)  # Divides by the total number of examples

    return average_loss, accuracy
