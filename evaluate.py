import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data import getLoaders
from model import TinyCNN
from config import MODEL_PATH


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, testLoader = getLoaders()

    model = TinyCNN(numClasses=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    allPreds = []
    allLabels = []

    with torch.no_grad():
        for imgs, labels in testLoader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            allPreds.extend(preds)
            allLabels.extend(labels.numpy())

    print("TEST ACCURACY:", accuracy_score(allLabels, allPreds))
    print("TEST F1:", f1_score(allLabels, allPreds))
    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(allLabels, allPreds, target_names=["real", "ai_generated"]))
    print("\nCONFUSION MATRIX:\n")
    print(confusion_matrix(allLabels, allPreds))


if __name__ == "__main__":
    main()