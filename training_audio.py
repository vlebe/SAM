from dataset import custom_collate_AudioDataset, train_test_split, AudioDataset
from audio_model import AudioMLPModel1, AudioMLPModel2, AudioMLPModel3
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np 

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
    dataset = AudioDataset('labels.csv', 'data/audio/samples/')
    train_dataset, val_dataset, test_dataset = train_test_split(dataset)
    train_dataloader, val_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_AudioDataset, num_workers=2, pin_memory=True), DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_AudioDataset, num_workers=1, pin_memory=True), DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_AudioDataset, num_workers=1, pin_memory=True)

    model = AudioMLPModel1()
    # model = AudioMLPModel2()
    # model = AudioMLPModel3()

    weight = torch.tensor([0.4, 0.6])
    # with torch.no_grad():
    #     total_label_1 = 0
    #     total_label_0 = 0
    #     for labels, _ in train_dataloader:
    #         total_label_1 += labels.sum()
    #         total_label_0 += (1-labels).sum()
    #     print(total_label_1, total_label_0)
    #     weight = torch.tensor([1 - total_label_0 / (total_label_0 + total_label_1), 1 - total_label_1 / (total_label_0 + total_label_1)])
    #     print(weight)
    
    criterion = torch.nn.CrossEntropyLoss(weight=weight.to('mps'))
    # criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    metrics = [f1_score, accuracy_score, precision_score, recall_score]

    training_loss = []
    validation_loss = []
    models_parameters = []
    non_valid_iteration = 0
    max_iter = 10
    for epoch in range(30):
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
        if non_valid_iteration == max_iter:
            print(f"Early stopping at epoch {epoch+1} : train loss {train_loss} valid loss {val_loss}")
            break
        else:
            print(f"Epoch {epoch} - Training Loss: {train_loss} - Validation Loss: {val_loss} - Validation Scores (F1 score, accuracy_score, precision score, recall score): {val_scores}")
        training_loss.append(train_loss)
        validation_loss.append(val_loss)
    pocket_model = AudioMLPModel1()
    pocket_model.load_state_dict(models_parameters[-1])
    

    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.legend()
    plt.savefig("data/audio_model_loss_test.png")

    _, test_score = evaluate(model, test_dataloader, criterion, metrics)
    _, pocket_test_score = evaluate(pocket_model, test_dataloader, criterion, metrics)
    print(f"Test Early Score: {test_score}")
    print(f"Test Pocket Score: {pocket_test_score}")

    # Save the model
    torch.save(model.state_dict(), "data/audio_model_Early_test.pt")
    torch.save(pocket_model.state_dict(), "data/audio_model_Early_pocket_test.pt")