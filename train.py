import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from config import EPOCHS, LEARNING_RATE, MODEL_PATH, SEED
from data import getLoaders
from model import TinyCNN


def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runEpoch(model, loader, lossFn, optimizer, device, trainMode=True):
    if trainMode:
        model.train()
    else:
        model.eval()

    totalLoss = 0
    allPreds = []
    allLabels = []

    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        if trainMode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(trainMode):
            outputs = model(imgs)
            loss = lossFn(outputs, labels)

            if trainMode:
                loss.backward()
                optimizer.step()

        totalLoss += loss.item() * imgs.size(0)

        preds = torch.argmax(outputs, dim=1)
        allPreds.extend(preds.cpu().numpy())
        allLabels.extend(labels.cpu().numpy())

    avgLoss = totalLoss / len(loader.dataset)
    acc = accuracy_score(allLabels, allPreds)
    f1 = f1_score(allLabels, allPreds)

    return avgLoss, acc, f1


def main():
    setSeed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE:", device)

    trainLoader, valLoader, _ = getLoaders()

    model = TinyCNN(numClasses=2).to(device)
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    bestValF1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch + 1}/{EPOCHS}")

        trainLoss, trainAcc, trainF1 = runEpoch(
            model, trainLoader, lossFn, optimizer, device, trainMode=True
        )

        valLoss, valAcc, valF1 = runEpoch(
            model, valLoader, lossFn, optimizer, device, trainMode=False
        )

        print(f"TRAIN LOSS: {trainLoss:.4f} | TRAIN ACC: {trainAcc:.4f} | TRAIN F1: {trainF1:.4f}")
        print(f"VAL LOSS:   {valLoss:.4f} | VAL ACC:   {valAcc:.4f} | VAL F1:   {valF1:.4f}")

        if valF1 > bestValF1:
            bestValF1 = valF1
            torch.save(model.state_dict(), MODEL_PATH)
            print("SAVED BEST MODEL")


if __name__ == "__main__":
    main()