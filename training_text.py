from dataset import Dataset, train_test_split, custom_collate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from text_model import TextModel, DistilCamembertEmbedding



def train(embedding_model, model, dataloader, criterion, optimizer, device):
    model.train()
    model.to(device)
    embedding_model.to("cpu")
    total_text_loss = 0
    for labels, txt, _, _ in tqdm(dataloader):
        labels = labels.type(torch.LongTensor)
        labels, input = labels.to(device), embedding_model(txt).to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_text_loss += loss.item()
    return total_text_loss / len(dataloader)

def evaluate(embedding_model, model, dataloader, criterion, metrics, device):
    model.eval()
    model.to(device)
    embedding_model.to("cpu")
    total_text_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for labels, txt, _, _ in tqdm(dataloader):
            labels = labels.type(torch.LongTensor)
            labels, input = labels.to(device), embedding_model(txt).to(device)
            outputs = model(input)
            loss = criterion(outputs, labels)
            total_text_loss += loss.item()
            for i, metric in enumerate(metrics):
                if metric.__name__ == "accuracy_score":
                    args = {'y_true': labels.cpu().numpy(), 
                        'y_pred': torch.argmax(outputs, dim=1).cpu().numpy()}
                else:
                    args = {'y_true': labels.cpu().numpy(), 
                            'y_pred': torch.argmax(outputs, dim=1).cpu().numpy(),
                            'zero_division': 0.0}
                scores[i] += metric(**args)
            scores = np.array(scores)
    return total_text_loss / len(dataloader), scores / len(dataloader)

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    embedding_model = DistilCamembertEmbedding()
    model = TextModel(768, 2)
    dataset = Dataset(labels_file_path="labels.csv", frame_dir="data/video/dataset_frame/", audio_dir="data/audio/samples/", txt_data_file_path="data.csv", img_shape=(3,256,256), mlp_audio=True)
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)
    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=2, pin_memory=True), DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=2, pin_memory=True), DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate, num_workers=2, pin_memory=True)
    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate), DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate), DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = [accuracy_score, f1_score, precision_score, recall_score]
    epochs = 10
    for epoch in range(epochs):
        train_loss = train(embedding_model, model, train_dataloader, criterion, optimizer, device)
        val_loss, val_scores = evaluate(embedding_model, model, val_dataloader, criterion, metrics, device)
        print("Epoch {} : Train Loss = {}, Val Loss = {}, Val Scores = {}".format(epoch, train_loss, val_loss, val_scores))
    test_loss, test_scores = evaluate(embedding_model, model, test_dataloader, criterion, metrics, device)
    print("Test Loss = {}, Test Scores = {}".format(test_loss, test_scores))

