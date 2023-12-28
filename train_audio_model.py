from audio_dataset import AudioDataset, custom_collate, train_test_split
from audio_model import AudioLSTMModel, AudioMLPModel

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for labels, mfcc in tqdm(dataloader):
        labels, mfcc = labels.to(device), mfcc.to(device)
        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, metric, device):
    model.eval()
    total_loss = 0
    score = 0
    with torch.no_grad():
        for labels, mfcc in dataloader:
            labels, mfcc = labels.to(device), mfcc.to(device)
            outputs = model(mfcc)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

    return total_loss / len(dataloader), score / len(dataloader)

if __name__ == "__main__":
    dataset = AudioDataset('data/audio/samples/', 'labels.csv', mlp=True)
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)
    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate), DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate), DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AudioLSTMModel().to(device)
    model = AudioMLPModel().to(device)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_loss = []
    validation_loss = []
    for epoch in range(30):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_score = evaluate(model, val_dataloader, criterion, f1_score ,device)
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val score: {val_score}")
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
    
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend()
    plt.savefig("audio_model_loss.png")

    _, test_score = evaluate(model, test_dataloader, criterion, f1_score, device)
    print(f"Test Score: {test_score}")

    # Save the model
    torch.save(model.state_dict(), "audio_model.pt")