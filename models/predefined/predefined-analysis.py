# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


# %% [markdown]
# # Using the wikiart dataset
#

# %%
train_image_location = "../../wikiart_train/"
test_image_location = "../../wikiart_test/"
results_location = "../../results/"

mean = pd.read_csv("../../data/wikiart_mean.csv", header=None).to_numpy()[0]
std = pd.read_csv("../../data/wikiart_std.csv", header=None).to_numpy()[0]

with open("../../data/num_styles.txt", "r") as f:
    num_classes = int(f.readline())
    print("Number of Styles:", num_classes)

print("Mean of dataset:", mean)
print("Std of dataset:", std)

has_mps = hasattr(torch, "has_mps") and torch.has_mps

train_df = pd.read_csv("../../data/wikiart_train.csv")
test_df = pd.read_csv("../../data/wikiart_test.csv")

weights = pd.read_csv("../../data/style_weights.csv").sort_values(by=["Style"])
weight_tensor = torch.tensor(weights["style_name"].values, dtype=torch.float32)

# Configuration
num_epochs = 15
filename_suffix = "_full_2"

# %%
# Check to see if mps works
cuda = torch.cuda.is_available()

device = None
if has_mps:
    print("MPS can be utilized")
    device = torch.device("mps")


# %% [markdown]
# ## Data Preprocessing
#
# - Downloading and extracting custom datasets
# - Loading custom datasets
# - Calculating the mean and std for normalization on custom datasets
# - Loading transforms to augment and normalize our data
#

# %%
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))["image"]


# Using albumentations library
train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256),
        A.SmallestMaxSize(max_size=256),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
        A.Flip(p=0.8),
        A.RandomCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256),
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

train = torchvision.datasets.ImageFolder(
    root=train_image_location, transform=Transforms(train_transform)
)
print("Number of train images", len(train.imgs))

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

test = torchvision.datasets.ImageFolder(
    root=test_image_location, transform=Transforms(test_transform)
)

test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

# %%
from tqdm import tqdm

# print('weight tensor', weight_tensor)
# criterion = nn.CrossEntropyLoss(weight=weight_tensor)
criterion = nn.CrossEntropyLoss()


def train_network(model, epochs):
    optimizer = optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)
    train_loss = [np.nan]
    train_accuracy = [np.nan]
    test_loss = [np.nan]

    model.train()

    pbar = tqdm(total=epochs, position=0, leave=True)
    pbar.set_description("EPOCH 1: Training Loss: NA, ")

    for epoch in range(epochs):

        total, correct, running_loss = 0, 0, 0

        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()

            if cuda:
                data, target = data.cuda(), target.cuda()
            elif has_mps:
                data = data.to(device)

            optimizer.zero_grad()

            out, _ = model(data)

            if has_mps:
                out = out.cpu()

            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            current_loss = loss.cpu().data.item()
            running_loss += current_loss

            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().cpu().item()

            pbar.set_description(
                f"EPOCH {epoch+1}\t Batch Loss: {current_loss:.3f}\t  Epoch Loss: {train_loss[-1]:.3f}\t Train Acc: {train_accuracy[-1]:.3f}\t Test Loss: {test_loss[-1]:.3f}\t"
            )

        test_running_loss = 0
        model.eval()

        with torch.no_grad():

            total_test, correct_test = 0, 0
            for batch_idx, (data, target) in enumerate(test_loader):

                if cuda:
                    data, target = data.cuda(), target.cuda()
                elif has_mps:
                    data = data.to(device)

                out, _ = model(data)

                if has_mps:
                    out = out.cpu()

                loss = criterion(out, target)

                test_running_loss += loss.cpu().data.item()

                _, predicted = out.max(1)
                total_test += target.size(0)
                correct_test += predicted.eq(target).sum().cpu().item()

            LTest = test_running_loss / len(test_loader)

        LTrain = running_loss / len(train_loader)
        accu = 100.0 * correct / total
        accu_test = 100.0 * correct_test / total_test

        train_accuracy.append(accu)
        train_loss.append(LTrain)
        test_loss.append(LTest)

        pbar.set_description(
            f"EPOCH {epoch+1}\t Batch Loss: {current_loss:.3f}\t  Epoch Loss: {train_loss[-1]:.3f}\t Train Acc: {train_accuracy[-1]:.3f}\t Test Loss: {test_loss[-1]:.3f}\t"
        )

        pbar.update()

    del train_accuracy[0]
    del train_loss[0]
    del test_loss[0]

    return train_accuracy, train_loss, test_loss


# %%
import seaborn as sns

from matplotlib import RcParams


def generate_train_test_loss(train_loss, test_loss, model_name):
    myrcparams = RcParams(
        {
            "axes.axisbelow": True,
            "axes.edgecolor": "white",
            "axes.facecolor": "#EAEAF2",
            "axes.grid": True,
            "axes.labelcolor": ".15",
            "axes.linewidth": 0.0,
            "figure.facecolor": "white",
            "font.family": ["serif"],
            "grid.color": "white",
            "grid.linestyle": "--",
            "image.cmap": "Greys",
            "legend.frameon": False,
            "legend.numpoints": 1,
            "legend.scatterpoints": 1,
            "lines.solid_capstyle": "round",
            "text.color": ".15",
            "xtick.color": ".15",
            "xtick.direction": "out",
            "xtick.major.size": 0.0,
            "xtick.minor.size": 0.0,
            "ytick.color": ".15",
            "ytick.direction": "out",
            "ytick.major.size": 0.0,
            "ytick.minor.size": 0.0,
        }
    )

    plt.style.library["seaborn-whitegrid"]
    RcParams.update(myrcparams)

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

    x = np.arange(1, len(train_loss) + 1)

    ax.plot(x, train_loss, "-o", label="Train Loss", linewidth=1.5)
    ax.plot(x, test_loss, "-o", label="Test Loss", linewidth=1.5)

    ax.set_xlabel("Epochs", fontsize=24)
    ax.set_ylabel("Loss", fontsize=24)

    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=20)
    ax.set_title("Loss vs Epochs", fontsize=24, fontweight="bold")

    plt.savefig(
        f"images/Loss vs Epochs({model_name}){filename_suffix}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"images/Loss vs Epochs({model_name}){filename_suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    new_loss_df = pd.DataFrame(
        {"Epochs": x.tolist(), "Training Loss": train_loss, "Testing Loss": test_loss}
    )
    new_loss_df["Model Name"] = model_name
    loss_results_filepath = f"state/loss_values{filename_suffix}.csv"

    print("Writing results to file")
    # If the dataframe exists
    if os.path.isfile(loss_results_filepath):
        loss_df = pd.read_csv(loss_results_filepath)

        # If the dataframe exists and the model has been run before
        if (loss_df[loss_df["Model Name"] == model_name]).any().any():
            loss_df = loss_df.drop(loss_df[loss_df["Model Name"] == model_name].index)

        loss_df = pd.concat([loss_df, new_loss_df], ignore_index=True, axis=0)
        loss_df.to_csv(loss_results_filepath, index=False)
    else:
        # Else write a new dataframe
        new_loss_df.to_csv(loss_results_filepath, index=False)


# %%
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# %%
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        targets = target.view(1, -1).expand_as(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


# %%
from sklearn.metrics import ConfusionMatrixDisplay


def calculate_accuracies(model):
    correct = []
    pred = []

    model.eval()

    with torch.no_grad():
        top1 = AverageMeter()
        top5 = AverageMeter()

        for batch_idx, (data, target) in enumerate(test_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            elif has_mps:
                data = data.to(device)

            out, _ = model(data)

            if has_mps:
                out = out.cpu()

            pred.extend(out.max(1)[1].tolist())
            correct.extend(target.tolist())

            prec1, prec5 = accuracy(out, target, topk=(1, 5))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

    print("Top 1% Average", top1.avg.item())
    print("Top 5% Average", top5.avg.item())

    new_results_df = pd.DataFrame(
        [
            {
                "Model Name": model.model_name,
                "Top 1%": top1.avg.item(),
                "Top 5%": top5.avg.item(),
            }
        ]
    )
    classification_results_filepath = (
        f"{results_location}/pretrained_classification_results{filename_suffix}.csv"
    )

    # Write the results to the file
    print("Writing results to file")
    # If the dataframe exists
    if os.path.isfile(classification_results_filepath):
        results_df = pd.read_csv(classification_results_filepath)

        # If the dataframe exists and the model has been run before
        if (results_df[results_df["Model Name"] == model.model_name]).any().any():
            results_df = results_df.drop(
                results_df[results_df["Model Name"] == model.model_name].index
            )

        results_df = pd.concat([results_df, new_results_df], ignore_index=True, axis=0)
        results_df.to_csv(classification_results_filepath, index=False)
    else:
        # Else write a new dataframe
        new_results_df.to_csv(classification_results_filepath, index=False)

    # Create a confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true=correct, y_pred=pred, display_labels=test.classes, ax=ax
    )
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
    ax.set_title(f"Confusion Matrix for {model.model_name}")
    fig.savefig(
        f"images/confusion_matrix_{model.model_name}{filename_suffix}.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    cm_results_df = pd.DataFrame(
        {
            "Correct": correct,
            "Predictions": pred,
            "Correct Label": [test.classes[idx] for idx in correct],
            "Prediction Label": [test.classes[idx] for idx in pred],
            "Image": [
                image_tuple[0].replace("../../wikiart_test/", "")
                for image_tuple in test.imgs
            ],
        }
    )
    cm_results_df["Model"] = model.model_name

    cm_results_df.to_csv(
        f"state/cm_results_{model.model_name}{filename_suffix}.csv", index=False
    )
    np.asarray(test.classes).tofile(f"state/classes{filename_suffix}.csv", sep=",")

    return top1.avg.item(), top5.avg.item()


# %%
def save_model(m, p):
    torch.save(m.state_dict(), p)


def load_model(m, p):
    m.load_state_dict(torch.load(p))


# %%
from torchvision import models
from torch import Tensor
from model_utils import (
    retrieve_models,
    PretrainedFeatureExtractor,
    PretrainedCustomResNet,
    FeatureExtractor,
)


feature_models = [
    # *retrieve_models(num_classes=num_classes),
    # PretrainedFeatureExtractor(
    #     model=models.inception_v3(weights="Inception_V3_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     layers=["Mixed_7c"],
    #     model_name="PretrainedCustomInceptionV3",
    #     classification_extraction=lambda x: x if isinstance(x, Tensor) else x.logits,
    # ),
    # PretrainedFeatureExtractor(
    #     model=models.googlenet(weights="GoogLeNet_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     layers=["inception5b"],
    #     model_name="PretrainedCustomGoogleNet",
    #     classification_extraction=lambda x: x if isinstance(x, Tensor) else x.logits,
    # ),
    # PretrainedFeatureExtractor(
    #     model=models.googlenet(weights="GoogLeNet_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     layers=["inception5b"],
    #     model_name="PretrainedCustomGoogleNet_LastLayer",
    #     classification_extraction=lambda x: x if isinstance(x, Tensor) else x.logits,
    #     filter_condition=lambda x: "inception5b" in x or "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet18(weights="ResNet18_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet18_FeatureOnly",
    #     filter_condition=lambda x: "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet18(weights="ResNet18_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet18_LastLayer",
    #     filter_condition=lambda x: "layer4" in x or "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet18(weights="ResNet18_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet18_Last2Layers",
    #     filter_condition=lambda x: "layer3" in x or "layer4" in x or "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet34(weights="ResNet34_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet34_FeatureOnly",
    #     filter_condition=lambda x: "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet34(weights="ResNet34_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet34_LastLayer",
    #     filter_condition=lambda x: "layer4" in x or "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet34(weights="ResNet34_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet34_Last2Layers",
    #     filter_condition=lambda x: "layer3" in x or "layer4" in x or "fc" in x,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet50(weights="ResNet50_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet50_FeatureOnly",
    #     filter_condition=lambda x: "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet50(weights="ResNet50_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet50_LastLayer",
    #     filter_condition=lambda x: "layer4" in x or "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet50(weights="ResNet50_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet50_Last2Layers",
    #     filter_condition=lambda x: "layer3" in x or "layer4" in x or "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet101(weights="ResNet101_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet101_FeatureOnly",
    #     filter_condition=lambda x: "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet101(weights="ResNet101_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet101_LastLayer",
    #     filter_condition=lambda x: "layer4" in x or "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet101(weights="ResNet101_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet101_Last2Layers",
    #     filter_condition=lambda x: "layer3" in x or "layer4" in x or "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet152(weights="ResNet152_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet152_FeatureOnly",
    #     filter_condition=lambda x: "fc" in x,
    #     batch_norm=2048,
    # ),
    # PretrainedCustomResNet(
    #     resnet=models.resnet152(weights="ResNet152_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     model_name="PretrainedCustomResNet152_LastLayer",
    #     filter_condition=lambda x: "layer4" in x or "fc" in x,
    #     batch_norm=2048,
    # ),
    PretrainedCustomResNet(
        resnet=models.resnet152(weights="ResNet152_Weights.DEFAULT"),
        num_classes=num_classes,
        model_name="PretrainedCustomResNet152_Last2Layers",
        filter_condition=lambda x: "layer3" in x or "layer4" in x or "fc" in x,
        batch_norm=2048,
    ),
    # PretrainedFeatureExtractor(
    #     model=models.googlenet(weights="GoogLeNet_Weights.DEFAULT"),
    #     num_classes=num_classes,
    #     layers=["inception5b"],
    #     model_name="PretrainedCustomGoogleNet_Last2Layers",
    #     classification_extraction=lambda x: x if isinstance(x, Tensor) else x.logits,
    #     filter_condition=lambda x: "inception5a" in x
    #     or "inception5b" in x
    #     or "fc" in x,
    # ),
]

print("Training", len(feature_models), "models")
print("Model names", [feature_model.model_name for feature_model in feature_models])
print("Number of epochs", num_epochs)

# %% [markdown]
# # Train and evaluate all of the networks

# %%
for feature_model in feature_models:
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    if cuda:
        feature_model.cuda()
    elif has_mps:
        feature_model = feature_model.to(device)

    print("Training the network", feature_model.model_name)
    train_accuracy, train_loss, test_loss = train_network(
        model=feature_model, epochs=num_epochs
    )

    p = f"state/{feature_model.model_name}_model{filename_suffix}.pth"
    save_model(feature_model, str(p))

    print("Generating train/test loss", feature_model.model_name)
    generate_train_test_loss(
        train_loss=train_loss, test_loss=test_loss, model_name=feature_model.model_name
    )

    print("Calculating accuracies", feature_model.model_name)
    top1, top5 = calculate_accuracies(model=feature_model)
