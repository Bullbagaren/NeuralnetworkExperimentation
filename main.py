import sys
import torch
import tqdm
from torch import nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
from NER_loader import ParquetDataset
from torch.utils.tensorboard import SummaryWriter, writer


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim=64, hidden_dim=128) -> None:
        super().__init__()
        self.relu_stack = nn.Sequential(
            nn.Embedding(vocab_size+1, embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_tags),
                )

    def forward(self, x):
        return self.relu_stack(x)


def main():
    training_path =  sys.argv[1]
    testing_path = sys.argv[2]

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    train_data = ParquetDataset(training_path)
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = ParquetDataset(testing_path)
    testloader = DataLoader(test_data, batch_size=64)

    model = NeuralNetwork(vocab_size=len(train_data.token2idx),
        num_tags=len(train_data.tag2idx)).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    writer = SummaryWriter("runs/token_tagger")

    epochs = 10
    for epoch in tqdm.tqdm(range(1,epochs+1), desc="Epochs"):
        train_loss = train(trainloader, model, loss_fn, optimizer, device, writer, epoch)
        test_loss, accuracy = test(testloader, model, loss_fn, device, writer, epoch)

        print(f"Epoch {epoch:>2} | train loss: {train_loss:.4f} | test loss: {test_loss:.4f} | acc: {accuracy*100:.1f}%")
    
    writer.close()


def train(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    model.train()
    epoch_loss = 0
    total_loss = 0
    loop = tqdm.tqdm(dataloader, desc=f"Train: {epoch}", leave=False)
    log_step = 10

    for batch, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoch_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4}")

        if batch % log_step == 0: 
            avg_loss = epoch_loss / log_step
            epoch_loss = 0
            writer.add_scalar("Loss/train", avg_loss, (batch+1)*epoch)
            #writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    return total_loss / len(dataloader)





def test(dataloader, model, loss_fn, device, writer, epoch):
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        loop = tqdm.tqdm(dataloader, desc=f"Test {epoch}", leave=False)
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    avg_loss = test_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    writer.add_scalar("Loss/test", avg_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)
    return avg_loss, accuracy





if __name__ == "__main__":
    main()
