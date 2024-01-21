from dataset import custom_collate_AudioDataset, train_test_split, AudioDataset
from model_audio import AudioMLPModel1, AudioMLPModel2, AudioMLPModel3, AudioRNNModel
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np 
import argparse
import os

class EnsembleModel1(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel1, self).__init__()
        self.models = models
        self.num_classes = 2

    def forward(self, x):
        with torch.no_grad():
            combined_tensor = torch.cat([model(x) for model in self.models], dim=0)
            sum_tensor = torch.sum(combined_tensor, dim=0)
            output = torch.argmax(sum_tensor)
        return output



class EnsembleModel2(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel2, self).__init__()
        self.models = models
        self.num_classes = 2
        self.weight = torch.nn.Parameter(torch.rand(1,self.num_classes))
        self.bias = torch.nn.Parameter(torch.rand(1,self.num_classes))

    def forward(self, x):
        with torch.no_grad():
            outputs = [model(x) for model in self.models]
            outputs = torch.stack(outputs)
            outputs = outputs * self.weight
            outputs = torch.sum(outputs, dim=0)
            outputs = outputs + self.bias
        return outputs
    


def train(model, dataloader, criterion, optimizer):
    model.train()
    device = torch.device('mps')
    model.to(device)
    total_loss = 0
    for labels, mfcc in tqdm(dataloader):
        labels = labels.type(torch.LongTensor)
        labels, mfcc = labels.to(device), mfcc.to(device)
        optimizer.zero_grad()
        outputs = model(mfcc)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, metrics):
    model.eval()
    device = torch.device('mps')
    model.to(device)
    total_loss = 0
    scores = [0 for _ in range(len(metrics))]
    with torch.no_grad():
        for labels, mfcc in tqdm(dataloader):
            labels = labels.type(torch.LongTensor)
            labels, mfcc = labels.to(device), mfcc.to(device)
            outputs = model(mfcc)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
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
    return total_loss / len(dataloader), scores / len(dataloader)

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum number of iteration, witout any improvement on the train loss with tol equal to 1e-3')
    parser.add_argument('--output_embedding_model_shape', type=tuple, default=(2048, 1, 1), help='output shape of the embedding model')
    parser.add_argument('--num_classes', type=int, default=2 , help='number of classes')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers, corresponding to number of CPU cores that want to be used for training and testing.')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model or not')
    parser.add_argument('--load_model', action='store_true', default=False, help='load model or not')
    parser.add_argument('--save_model_path', type=str, default='data/ModelAudio/Models/', help='path to save model')
    args = parser.parse_args()
    

    dataset = AudioDataset('labels.csv', 'data/audio/samples/', mlp_audio=False)
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_AudioDataset, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_AudioDataset, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_AudioDataset, num_workers=1, pin_memory=True)

    # model = AudioMLPModel1()
    # model = AudioMLPModel2()
    # model = AudioMLPModel3()
    model = AudioRNNModel()
    weight = torch.tensor([0.2, 0.8])
    criterion = torch.nn.CrossEntropyLoss(weight=weight.to(device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics = [f1_score, accuracy_score, precision_score, recall_score]

    if args.save_model:
        training_loss = []
        validation_loss = []
        models_parameters = []
        non_valid_iteration = 0
        for epoch in range(args.num_epochs):
            train_loss = train(model, train_dataloader, criterion, optimizer)
            val_loss, val_scores = evaluate(model, val_dataloader, criterion, metrics)
            if epoch == 0:
                min_loss = val_loss
            else:
                min_loss = min(validation_loss)
            if val_loss < min_loss:
                models_parameters.append(model.state_dict())
                non_valid_iteration = 0
            else:
                non_valid_iteration += 1
            if non_valid_iteration == args.max_iter:
                print(f"Early stopping at epoch {epoch+1} : train loss {train_loss} valid loss {val_loss}")
                break
            else:
                print(f"Epoch {epoch} - Training Loss: {train_loss} - Validation Loss: {val_loss} - Validation Scores (F1 score, accuracy_score, precision score, recall score): {val_scores}")
            training_loss.append(train_loss)
            validation_loss.append(val_loss)

        if len(models_parameters) != 0:
            pocket_model = AudioRNNModel()
            pocket_model.load_state_dict(models_parameters[-1])
            torch.save(pocket_model.state_dict(), "data/ModelAudio/Models/audio_model_Early_pocket_RNN_V1.pt")
            _, pocket_test_score = evaluate(pocket_model, test_dataloader, criterion, metrics)
            print(f"Test Pocket Score: {pocket_test_score}")
        
        plt.plot(training_loss, label="Training Loss")
        plt.plot(validation_loss, label="Validation Loss")
        plt.legend()
        plt.savefig("data/ModelAudio/Graphs/audio_model_loss_RNN_V1.png")
        torch.save(model.state_dict(), "data/ModelAudio/Models/audio_model_Early_RNN_V1.pt")
        _, test_score = evaluate(model, test_dataloader, criterion, metrics)
        print(f"Test Early Score: {test_score}")

    if args.load_model:
        model.load_state_dict(torch.load("data/ModelAudio/Models/audio_model_Early_RNN_V1.pt"))
        _, test_score = evaluate(model, test_dataloader, criterion, metrics)
        print(f"Test Early Score: {test_score}")
        pocket_model = AudioRNNModel()
        pocket_model.load_state_dict(torch.load("data/ModelAudio/Models/audio_model_Early_pocket_RNN_V1.pt"))
        _, pocket_test_score = evaluate(pocket_model, test_dataloader, criterion, metrics)
        print(f"Test Pocket Score: {pocket_test_score}")

