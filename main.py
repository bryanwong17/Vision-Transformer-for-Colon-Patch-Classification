import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import timm
import pickle
import copy
import os
import random
from datetime import datetime

from sklearn.metrics import accuracy_score
from sklearn import model_selection

from draw import get_loss_curve, get_accuracy_curve, get_confusion_matrix, get_roc_curve
from dataset import PatchesDataset

from torch.autograd import Variable
from pytorchtools import EarlyStopping

from models.vit import ViT

from mixup import mixup_data, mixup_criterion

parser = argparse.ArgumentParser(description="Colon Patch Classification")
parser.add_argument("--model", default="vit_base_patch16_384", type=str,
                    help="name of model")
parser.add_argument("--num_classes", default=3, type=int,
                    help="target classification classes")
parser.add_argument("--mode", default="train", type=str,
                    help="train or test")
parser.add_argument("--train_ratio", default=0.85, type=float,
                    help="train data ratio")
parser.add_argument("--save_folder_path", default="vit_base_patch16_384/big_data/pretrained_mixup(0.2)", type=str,
                    help="save folder path")
parser.add_argument("--batch_size", default=8, type=int,
                    help="batch size")
parser.add_argument("--optim", default="sgd", type=str,
                    help="optimizer")
parser.add_argument("--lr", default=2e-5, type=float,
                    help="learning rate")
parser.add_argument('--decay', default=2e-6, type=float,
                    help='weight decay')
parser.add_argument("--epochs", default=100, type=int,
                    help="train epoch")
parser.add_argument('--alpha', default=0.2, type=float,
                    help='mixup interpolation coefficient (default: 1.0)')

args = parser.parse_args()


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    # keep track of training loss
    epoch_loss = 0.0
    loss = 0
    total = 0
    correct = 0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if device.type == "cuda":
            data, target = data.cuda(), target.cuda()

        data, target_a, target_b, lam = mixup_data(data, target, args.alpha, use_cuda=True)
        data, target_a, target_b = map(Variable, (data, target_a, target_b))
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        epoch_loss += loss
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # update parameters
        optimizer.step()

    return epoch_loss / len(train_loader), 100. * correct / total

def validate_one_epoch(model, valid_loader, criterion, device):
    # keep track of validation loss
    valid_loss = 0.0
    loss = 0
    total = 0
    correct = 0
    ######################
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is 
        # available
        if device.type == "cuda":
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            valid_loss += loss
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

    return valid_loss / len(valid_loader), 100. * correct / total

def test_model(model, test_loader, device):
    # keep track list of predicted and actual for confusion matrix
    label_predicted = []
    label_actual = []
    temp_actual = []
    predicted_score = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if device.type == "cuda":
                images, labels = images.cuda(), labels.cuda()
            output = model(images) # batch size(8) x number of classes(3)
            # the class with the highest energy is what we choose as prediction
            _ , predicted = torch.max(output.data, 1)
            # for loop one batch
            for x in predicted:
                label_predicted.append(x.item())
            for y in labels:
                label_actual.append(y.item())
                temp_actual.append(y.item())

            # get the predicted score based on label class
            count = 0
            # every row sum is 1
            output_sigmoid = torch.softmax(output, dim=1)
            for item in output_sigmoid:
                predicted_score.append(item[temp_actual[count]].item())
                count = count + 1

            temp_actual.clear()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return label_actual, label_predicted, predicted_score


def fit_gpu(model, save_folder_path, epochs, device, criterion, optimizer, train_loader, valid_loader=None):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    valid_loss_min = 100000 # track change in validation loss

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []

    # early stopping
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(1, epochs + 1):

        # train_loader
        print(f"{'='*50}")
        print(f"EPOCH {epoch} - TRAINING...")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"\t[TRAIN] LOSS: {train_loss}, ACCURACY: {train_acc}\n")

        train_losses.append(train_loss.item())
        train_accs.append(train_acc.item())
       
        # valid_loader
        if valid_loader is not None:
            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc = validate_one_epoch(
                model, valid_loader, criterion, device
            )
            print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")

            valid_losses.append(valid_loss.item())
            valid_accs.append(valid_acc.item())

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and epoch != 1:
                valid_loss_min = valid_loss
                # deep copy the model
                best_model_wts = copy.deepcopy(model.state_dict())

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "criterion": criterion,
            "optimizer": optimizer
        }

        # write train val loss and acc to pickle
        save_log_path = save_folder_path + "/log"
        if not os.path.exists(save_log_path):
            os.makedirs(save_log_path)
        with open(save_log_path + f"/log_{epoch}.pkl","wb") as f:
            pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)

        # save model for every epoch
        save_model_path = save_folder_path + "/model"
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        torch.save(model.state_dict(), save_model_path + f"/model_{epoch}.pth")

        # early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_losses, valid_losses, train_accs, valid_accs

def main():

    # for reproductibility
    seed_everything(1001)
   
    # train or test mode for data preprocessing
    if args.mode == "train":
        df = pd.read_csv("big_data/train/train.csv")
        # valid_df = pd.read_csv("big_data/valid.csv")
        train_df, valid_df = model_selection.train_test_split(
            df, test_size=0.15, random_state=42, stratify=df.label.values
        )

        # train_df.size = number of rows * columns
        # number of rows = size / column(2)

        train_dataset = PatchesDataset(train_df, mode="train")
        valid_dataset = PatchesDataset(valid_df, mode="valid")

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    else:
        test_df = pd.read_csv("big_data/test/test.csv")

        test_dataset = PatchesDataset(test_df, mode="test")

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

    # define architecture
    if args.model == "vit_base_patch16_384":
        # create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # ImageNet-1k weights fine-tuned from imageNet-21k
        # patch_size=16, embed_dim=768, depth=12, num_heads=12
        model = timm.create_model("vit_base_patch16_384", pretrained=True)
        # print(model.head.in_features) # 768
        model.head = nn.Linear(model.head.in_features, args.num_classes)
        # model.classifier = nn.Sequential(nn.Linear(model.head.in_features, 512),
        #                    nn.ReLU(),
        #                    nn.Linear(512, args.num_classes),
        #                    nn.Softmax(dim=1))
    elif args.model == "vit_base_patch16_384_scratch":
        model = ViT(image_size=384, patch_size=16, num_classes=3, dim=768, depth=12, heads=12, mlp_dim=3072)
    elif args.model == "vit_large_patch16_224":
        # ImageNet-1k weights fine-tuned from imageNet-21k
        # patch_size=16, embed_dim=1024, depth=24, num_heads=16
        model = timm.create_model("vit_large_patch16_224", pretrained=True)
        model.head = nn.Linear(model.head.in_features, args.num_classes)
    elif args.model == "vit_small_patch16_384":
        # ImageNet-1k weights fine-tuned from imageNet-21k
        # patch_size=16, embed_dim=384, depth=12, num_heads=6
        model = timm.create_model("vit_small_patch16_384", pretrained=True)
        model.head = nn.Linear(model.head.in_features, args.num_classes)
    elif args.model == "densenet201":
        model = timm.create_model("densenet201", pretrained=True)
        model.classifier.out_features = args.num_classes
    else:
        # it will be added soon
        pass

    # define criterion
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # select optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = datetime.now()
    print("Start Time: {start_time}")

    if args.mode == "train":
        best_model, train_losses, valid_losses, train_accs, valid_accs = fit_gpu(
            model=model,
            save_folder_path=args.save_folder_path,
            epochs=args.epochs,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader
        )

        print(f"Training Execution time: {datetime.now() - start_time}")
        print("Saving Model")
        save_best_model_path = args.save_folder_path + "/model"
        torch.save(best_model.state_dict(), save_best_model_path + "/best_model.pth")

        save_figures_path = args.save_folder_path + "/figures"
        if not os.path.exists(save_figures_path):
            os.makedirs(save_figures_path)

        # plot train and valid loss
        get_loss_curve(save_figures_path, train_losses, valid_losses)

        # plot train and valid acc
        get_accuracy_curve(save_figures_path, train_accs, valid_accs)

    elif args.mode == "test":
        save_best_model_path = args.save_folder_path + "/model/best_model.pth"
        model.load_state_dict(torch.load(save_best_model_path))
        label_actual, label_predicted, predicted_score = test_model(model, test_loader, device)
        print(f"Testing Execution time: {datetime.now() - start_time}")

        # accuracy score
        print("Accuracy Score:")
        print(accuracy_score(label_actual, label_predicted) * 100)

        save_figures_path = args.save_folder_path + "/figures"
        if not os.path.exists(save_figures_path):
            os.makedirs(save_figures_path)

        # roc curve
        get_roc_curve(save_figures_path, label_actual, predicted_score)
        # confusion matrix
        get_confusion_matrix(save_figures_path, label_actual, label_predicted)

# if this file is run directly by python
if __name__ == "__main__":
    main()





