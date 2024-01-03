from audio_dataset import train_test_split #, AudioDataset
from final_vggish_dataset import AudioDataset
import matplotlib.pyplot as plt
import torch
from vggish_dataloader import CustomDataLoader
from tqdm import tqdm
from vggish import VGGish
from sklearn.metrics import f1_score, precision_score, recall_score 

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    accuracy = 0
    all_preds = []
    all_labels = []

    for labels, mfcc in tqdm(dataloader):
        labels, mfcc = labels.to(device), mfcc.to(device)
        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        all_labels.append(labels.item())
        all_preds.append(outputs.item() >= 0.5)
        if outputs.item() >= 0.5 and labels.item() == 1:
            accuracy += 1
        if outputs.item() < 0.5 and labels.item() == 0:
            accuracy += 1
        total_loss += loss.item()
    return total_loss / len(dataloader), accuracy / len(dataloader), f1_score(all_labels, all_preds), precision_score(all_labels, all_preds), recall_score(all_labels, all_preds)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    score = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for labels, mfcc in dataloader:
            labels, mfcc = labels.to(device), mfcc.to(device)
            outputs = model(mfcc)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            if outputs.item() >= 0.5 and labels.item() == 1:
                score += 1
            if outputs.item() < 0.5 and labels.item() == 0:
                score += 1
            all_labels.append(labels.item())
            all_preds.append(outputs.item() >= 0.5)

    return total_loss / len(dataloader), score / len(dataloader), f1_score(all_labels, all_preds), precision_score(all_labels, all_preds), recall_score(all_labels, all_preds)

if __name__ == "__main__":
    dataset = AudioDataset('data/audio/samples/', 'labels.csv')
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)
    train_dataloader, val_dataloader, test_dataloader = CustomDataLoader(train_dataset, batch_size=1, shuffle=True), CustomDataLoader(val_dataset, batch_size=1, shuffle=True), CustomDataLoader(test_dataset, batch_size=1, shuffle=True)   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGish().to(device)

    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_loss = []
    validation_loss = []
    for epoch in range(30):
        train_loss, train_acc, train_f1, train_prec, train_rec = train(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_prec, val_rec = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Train acc : {train_acc}/{train_f1}/{train_prec}/{train_rec}, Val Loss: {val_loss}, Val score: {val_acc}/{val_f1}/{val_prec}/{val_rec}")
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