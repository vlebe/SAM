from dataset import TextDataset, train_test_split
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from text_model import TextModel, DistilCamembertEmbedding
from transformers import AutoTokenizer
import matplotlib.pyplot as plt



def train(embedding_model, model, dataloader, criterion, optimizer, device):
    print("Training...")
    global tokenizer
    model.train()
    embedding_model.train()
    total_text_loss = 0
    for batch in tqdm(dataloader):
        labels, txt = batch
        if len(labels) == 1:
            continue
        labels = labels.to(device)
        tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
        with torch.no_grad():
            txt = embedding_model(input_ids, attention_mask)
        optimizer.zero_grad()
        outputs = model(txt)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_text_loss += loss.item()
    return total_text_loss / len(dataloader)

def evaluate(embedding_model, model, dataloader, criterion, metrics, device):
    print("Evaluating...")
    global tokenizer
    model.eval()
    embedding_model.eval()
    total_text_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for labels, txt in tqdm(dataloader):
            labels = labels.to(device)
            tokenizer_output = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids, attention_mask = tokenizer_output["input_ids"].to(device), tokenizer_output["attention_mask"].to(device)
            txt = embedding_model(input_ids, attention_mask)
            outputs = model(txt)
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

    model = TextModel(768, 2).to(device)
    embedding_model = DistilCamembertEmbedding().to(device)
    batch_size = 32
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=False)


    dataset = TextDataset('labels.csv', 'txt_data.csv')
    valid_ratio = 0.1
    print(valid_ratio)
    test_ratio = 0.1
    print(test_ratio)
    train_dataset, validation_dataset, test_dataset = train_test_split(dataset, val_size=valid_ratio, test_size=test_ratio)
    workers = True
    if (device == torch.device('cuda') or device == torch.device('mps')) and workers:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_dataset, shuffle=True, num_workers=1, pin_memory=True)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=True)

    
    weights = torch.tensor([0.1, 0.9])
    weights = weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = [accuracy_score, f1_score, precision_score, recall_score]
    epochs = 100
    models_parameters = []
    early_stopping = False
    min_val_loss = float('inf')
    train_losses = []
    val_losses = []
    max_iter = 20
    iter_non_valid = 0
    # for epoch in range(epochs):
    #     if early_stopping:
    #         print("Early stopping")
    #         break
    #     if not(early_stopping):
    #         train_loss = train(embedding_model, model, train_loader, criterion, optimizer, device)
    #         train_losses.append(train_loss)
    #         val_loss, val_scores = evaluate(embedding_model, model, validation_loader, criterion, metrics, device)
    #         val_losses.append(val_loss)
    #         if epoch == 0:
    #             min_val_loss = val_loss
    #         elif val_loss < min_val_loss:
    #             min_val_loss = val_loss
    #             iter_non_valid = 0
    #             models_parameters.append(model.state_dict())
    #         else:
    #             iter_non_valid += 1
    #             if iter_non_valid == max_iter:
    #                 early_stopping = True
    #     print("Epoch {} : Train Loss = {}, Val Loss = {}, Val Scores = {}".format(epoch, train_loss, val_loss, val_scores))

    # plt.plot(train_losses, label="Training Loss")
    # plt.plot(val_losses, label="Validation Loss")
    # plt.legend()
    # plt.savefig("data/text_model_loss.png")
    # torch.save(model.state_dict(), "text_model_Early.pt")
    # torch.save(models_parameters[-1], "text_model_Early_Pocket_Algo.pt")
    # pocket_model = TextModel(768, 2).to(device)
    # pocket_model.load_state_dict(models_parameters[-1])
    model.load_state_dict(torch.load("model_text.pt"))
    test_loss, test_scores = evaluate(embedding_model, model, test_loader, criterion, metrics, device)
    # pocket_test_loss, pocket_test_scores = evaluate(embedding_model, pocket_model, test_loader, criterion, metrics, device)
    print("Test Loss = {}, Test Scores = {}".format(test_loss, test_scores))
    # print("Pocket Test Loss = {}, Pocket Test Scores = {}".format(pocket_test_loss, pocket_test_scores))

