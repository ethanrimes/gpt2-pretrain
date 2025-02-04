import torch
from torch.utils.data import DataLoader, TensorDataset

def evaluate_perplexity(model, test_data, device, batch_size):
    """
    Manually calculate the perplexity of the model using the test data.
    """
    model.eval()  # Set model to evaluation mode
    test_dataset = TensorDataset(test_data["input_ids"].to(device))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch[0].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            print(f"Loss: {loss.item()}")
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Average Loss: {avg_loss}")
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Perplexity: {perplexity.item()}")
    return perplexity.item()
