from dataset import TextDataset, train_test_split
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from model_text import TextModel, TextModel2, TextModel3, DistilCamembertEmbedding, FlaubertEmbedding
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import argparse



def train(embedding_model, model, dataloader, criterion, optimizer, device):
    print("Training...")
    global tokenizer
    model.train()
    embedding_model.eval()
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.00007, help='learning rate')
    parser.add_argument('--max_iter', type=int, default=20, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=int, default=768, help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing. 6 is recommended if available.')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model or not')
    parser.add_argument('--load_model', action='store_true', default=False, help='load model or not')
    parser.add_argument('--save_model_path', type=str, default='data/ModelText/Models/', help='path to save model')
    args = parser.parse_args()

    model = TextModel2(args.output_embedding_model_shape, args.num_classes).to(device)
    # model = TextModel3(args.output_embedding_model_shape, args.num_classes).to(device)
    embedding_model = DistilCamembertEmbedding().to(device)
    # embedding_model = FlaubertEmbedding().to(device)
    tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base", use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    dataset = TextDataset('labels.csv', 'txt_data.csv')


    labels = dataset.labels["turn_after"].values
    #Doing the same but by using subset and indices
    class_0_indices = [i for i in range(len(dataset)) if labels[i] == 0]
    class_1_indices = [i for i in range(len(dataset)) if labels[i] == 1]
    #subsampling randomly class 0 to have the same number of samples as class 1
    subsampled_indices_0 = np.random.choice(class_0_indices, len(class_1_indices), replace=False)
    subsampled_indices = subsampled_indices_0.tolist() + class_1_indices
    subdataset = torch.utils.data.Subset(dataset, subsampled_indices)
    train_dataset, validation_dataset, test_dataset = train_test_split(subdataset, test_size=0.10, val_size=0.15)
    if device == torch.device('cuda') or device == torch.device('mps'):
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_dataset, shuffle=True, num_workers=1, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=True)

    print(len(train_loader))
    # weights = torch.tensor([1.0, 1.0])
    weights = torch.tensor([0.1, 0.9])
    weights = weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics = [accuracy_score, f1_score, precision_score, recall_score]

    if args.save_model:
        models_parameters = []
        early_stopping = False
        min_val_loss = float('inf')
        train_losses = []
        val_losses = []
        iter_non_valid = 0
        for epoch in range(args.num_epochs):
            if early_stopping:
                print("Early stopping")
                break
            if not(early_stopping):
                train_loss = train(embedding_model, model, train_loader, criterion, optimizer, device)
                train_losses.append(train_loss)
                val_loss, val_scores = evaluate(embedding_model, model, validation_loader, criterion, metrics, device)
                val_losses.append(val_loss)
                if epoch == 0:
                    min_val_loss = val_loss
                elif val_loss < min_val_loss:
                    min_val_loss = val_loss
                    iter_non_valid = 0
                    models_parameters.append(model.state_dict())
                else:
                    iter_non_valid += 1
                    if iter_non_valid == args.max_iter:
                        early_stopping = True
            print("Epoch {} : Train Loss = {}, Val Loss = {}, Val Scores = {}".format(epoch, train_loss, val_loss, val_scores))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.savefig("data/ModelText/Graphs/text_model_lossV2.2.png")
        torch.save(model.state_dict(), "data/ModelText/Models/text_model_EarlyV2.2.pt")
        test_loss, test_scores = evaluate(embedding_model, model, test_loader, criterion, metrics, device)
        print("Test Loss = {}, Test Scores = {}".format(test_loss, test_scores))
        if len(models_parameters) > 0:
            pocket_model = TextModel2(args.output_embedding_model_shape, args.num_classes).to(device)
            pocket_model.load_state_dict(models_parameters[-1])
            torch.save(models_parameters[-1], "data/ModelText/Models/text_model_Early_Pocket_AlgoV2.2.pt")
            pocket_test_loss, pocket_test_scores = evaluate(embedding_model, pocket_model, test_loader, criterion, metrics, device) 
            print("Pocket Test Loss = {}, Pocket Test Scores = {}".format(pocket_test_loss, pocket_test_scores))
    
    if args.load_model:
        model.load_state_dict(torch.load("data/ModelText/Models/text_model_Early.pt"))
        test_loss, test_scores = evaluate(embedding_model, model, test_loader, criterion, metrics, device)
        print("Test Loss = {}, Test Scores = {}".format(test_loss, test_scores))
        pocket_model = TextModel2(args.output_embedding_model_shape, args.num_classes).to(device)
        pocket_model.load_state_dict(torch.load("data/ModelText/Models/text_model_Early_Pocket_Algo.pt"))
        pocket_test_loss, pocket_test_scores = evaluate(embedding_model, pocket_model, test_loader, criterion, metrics, device) 
        print("Pocket Test Loss = {}, Pocket Test Scores = {}".format(pocket_test_loss, pocket_test_scores))
        

