import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
import argparse
import time
import copy
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

class Embeddings(torch.utils.data.Dataset):
    def __init__(self, labelfile, datadir, maxsize,lab):
        self.labels = pd.read_csv(labelfile)
        self.datadir = datadir
        self.maxsize = maxsize
        self.lab = lab

    def __getitem__(self, index):
        data = np.load(self.datadir + str(index) + ".npy")
        label = torch.FloatTensor([1 if data[1] == self.lab else 0]) # logic to convert all classes(except 1) to 0
        inputembed = torch.from_numpy(data[0]).float()
        seq_length = inputembed.size()[0]
        pad_size = self.maxsize - seq_length
        if pad_size:
            inputembed = torch.cat([inputembed, torch.zeros(pad_size,inputembed.size()[1])])
        return {"embeddings": inputembed, "label": label, "seq_length": seq_length}

    def __len__(self):
        return len(self.labels)

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, batch_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2label = nn.Linear(hidden_dim, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, seq_lengths):
        batch_size, seq_len, _ = x.size()

        packed_input = pack_padded_sequence(x, seq_lengths.cpu().numpy(), batch_first=True)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(device)
        lstm_out, (h0, c0) = self.lstm(packed_input, (h0, c0))
        out_rnn, ht = pad_packed_sequence(lstm_out, batch_first=True)
        out = self.hidden2label(out_rnn)
        batchindex = [i for i in range(batch_size)]
        columnindex = seq_lengths - 1
        last_tensor = out[batchindex, columnindex, :]
        out_sig = self.sigmoid(last_tensor)
        return out_sig

def sort_batch(batch):
    data = pad_sequence([i['embeddings'] for i in batch], batch_first=True)
    seq_length = torch.Tensor([i['seq_length'] for i in batch])
    target = torch.FloatTensor([i['label'] for i in batch])
    sortedbatch = sorted([[index, value] for index, value in enumerate(seq_length)], key=lambda x: x[1],
                         reverse=True)
    order = [i[0] for i in sortedbatch]
    sdata, starget = [], []
    for index in order:
        sdata.append(data[index])
        starget.append(target[index])
    sdata = torch.stack(sdata)
    starget = torch.stack(starget)
    return {'embeddings': sdata, 'label': starget, 'seq_length': torch.LongTensor([i[1] for i in sortedbatch])}


def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes, weight_decay, lr, batch_size):

    since = time.time()
    best_model_ets = copy.deepcopy(model.state_dict())
    best_loss = 10000
    best_epoch = -1
    best_lr = -1

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('='*100)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            nbatches = int(dataset_sizes[phase]/batch_size)
            for i, sample in enumerate(dataloaders[phase]):
                inputs = sample['embeddings'].to(device)
                labels = sample['label'].to(device)
                seq_length = sample['seq_length'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs, seq_length)
                    trans_y = (labels.cpu().numpy() == 1).astype(int)  # change y to binary [1 or 0], 1 if value == 1 else 0

                    # set weights to total_number_of_samples / nr_samples_of_class_i
                    num_class_i = sum(trans_y)
                    weights = [len(trans_y) / num_class_i if x == 1 else len(trans_y) / (len(trans_y) - num_class_i) for
                               x in trans_y]

                    criterion.weight = torch.FloatTensor(weights).to(device)

                    loss = criterion.forward(outputs,  torch.Tensor(trans_y).to(device))  # criteria should be binary cross entropy

                    if phase == "train":
                        loss.backward() #backprop
                        optimizer.step() #weights are updated

                running_loss += loss.item() * batch_size
                print("Batch: {}/{}, Loss: {}".format(i, nbatches, running_loss/((i+1) * batch_size)))
            epoch_loss = running_loss / dataset_sizes[phase]
            print("Phase: {}, Epoch: {}, loss: {:4f}".format(phase,epoch, epoch_loss))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_lr = lr
                best_model_ets = copy.deepcopy(model.state_dict())

            if phase == 'val' and epoch_loss > best_loss:
                lr = lr/10

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                    )
                print("New optimizer created with LR: {}".format(lr))

        if epoch - best_epoch > 3:
            print("Early termination of epoch - no improvements in the last 3 epochs")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    print('Best Model: {}'.format(best_model_ets))
    model.load_state_dict(best_model_ets)

    return model, best_epoch, best_loss, best_lr


def test_model(model, testloader, dataset_sizes, batch_size):

    preds = []
    actual = []
    nbatches = int(dataset_sizes['test'] / batch_size)
    for i, sample in enumerate(testloader):
        inputs = sample['embeddings'].to(device)
        labels = sample['label'].to(device)
        seq_length = sample['seq_length'].to(device)
        output = model(inputs, seq_length)
        actual.extend(labels.cpu().numpy())
        preds.extend(output.cpu().data.numpy())
        print("Batch: {}/{}".format(i, nbatches))
    return actual, preds


def generate_ROC_AUC_curve(act, pred, name, n_classes):
    fpr, tpr, roc_auc = {}, {}, {}
    for n in range(n_classes):
        actn , predn = [], []
        for index, value in enumerate(act):
            actn.append(value[n])
            predn.append(pred[index][n])

        fpr[n], tpr[n], _ = roc_curve(actn, predn)
        roc_auc[n] = auc(fpr[n], tpr[n])

        try:
            print('AUC for {} is {}'.format(n, roc_auc_score(actn, predn)))
        except:
            print('cannot calculate AUC for {}'.format(n))

    plt.figure()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Baseline")
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'black', 'navy'])
    for i, color in zip([i for i in range(n_classes)], colors):
        plt.plot(fpr[i], tpr[i], color=color, linestyle='--',
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig("/home/t/tanjw/results/ROC_{}.png".format(name), bbox_inches='tight')

    print(roc_auc)
    return roc_auc

def generate_ROC_AUC_curve_single(act, pred, dir, name):
    actn , predn = [], []
    for index, value in enumerate(act):
        actn.append(value)
        predn.append(pred[index])

    fpr, tpr, _ = roc_curve(actn, predn)
    roc_auc = auc(fpr, tpr)

    try:
        print('AUC: '.format(roc_auc_score(actn, predn)))
    except:
        print('cannot calculate AUC')

    plt.figure()
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="Baseline")
    plt.plot(fpr, tpr, color='aqua', linestyle='--',
             label='Area: {}'.format(roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig(dir + "ROC_{}.png".format(name), bbox_inches='tight')

    print(roc_auc)

    precision, recall, average_precision, auc_scores = [], [], [], []
    precision, recall, _ = precision_recall_curve(act, pred)
    average_precision = average_precision_score(act, pred, average='weighted')
    print('Average Precision is {}'.format(average_precision))

    lines = []
    labels = []
    plt.figure()
    plt.style.use('ggplot')

    l, = plt.plot(recall, precision, color='blue', lw=2)
    lines.append(l)
    labels.append("AP: {:0.2f}".format(average_precision))

    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(lines, labels, title="PR-AUC", loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.savefig(dir+"PR_{}.png".format(name), bbox_inches='tight')

    print('Precision Recall Curve generated!')

    return roc_auc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Name of model")
    parser.add_argument('--dd', help="filepath of directory where data is stored")
    parser.add_argument('--rd', help="filepath of directory where results/plots will be store")
    parser.add_argument('--tr', help="name of train csv file")
    parser.add_argument('--te', help="name of test csv file")
    parser.add_argument('--ev', help="name of validation csv file")
    args = parser.parse_args()

    datadirectory = args.dd
    resultsdirectory = args.rd

    print("Cuda Available:", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight_decay = 1e-4
    lr = 0.1
    num_classes = 1
    batch_size = 64
    num_epochs = 20
    num_layers = 16


    train = Embeddings(labelfile=datadirectory + args.tr, datadir=datadirectory + "train/", maxsize=512,
                       lab=1)
    val = Embeddings(labelfile=datadirectory + args.ev, datadir=datadirectory + "val/", maxsize=512, lab=1)
    test = Embeddings(labelfile=datadirectory + args.te, datadir=datadirectory + "test/", maxsize=512, lab=1)

    trainloader = torch.utils.data.DataLoader(train,
                                               batch_size=batch_size,
                                              collate_fn=sort_batch,
                                              num_workers=10)

    validloader = torch.utils.data.DataLoader(val,
                                               batch_size=batch_size,
                                              collate_fn=sort_batch,
                                              num_workers=10)

    testloader = torch.utils.data.DataLoader(test,
                                              batch_size=batch_size,
                                             collate_fn=sort_batch,
                                             num_workers=10)


    train_val_loader = {'train': trainloader, 'val': validloader}

    dataset_sizes = {'train': train.__len__(), 'val': val.__len__(), 'test': test.__len__()}

    model = LSTMClassifier(embedding_dim=768, hidden_dim=64, output_size=1, batch_size=batch_size, num_layers=num_layers)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay=weight_decay, momentum=0.9)

    model, best_epoch, best_loss, best_lr = train_model(model, criterion, optimizer, num_epochs, train_val_loader, dataset_sizes, weight_decay, lr, batch_size)
    print("Best Epoch: {}, loss: {}, learning rate: {}".format(best_epoch, best_loss, best_lr))
    print("Testing model.......")
    actual, prediction = test_model(model, testloader, dataset_sizes, batch_size)
    out = pd.DataFrame({'actual': actual, 'prediction':prediction})
    out.to_csv("/home/t/tanjw/results/bertresults.csv",index=False)
    generate_ROC_AUC_curve_single(actual, prediction, resultsdirectory, "_class_".format(str(1)))