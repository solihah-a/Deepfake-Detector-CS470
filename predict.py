import torch
from PIL import Image
from torchvision import transforms

from model import TinyCNN
from config import MODEL_PATH, IMAGE_SIZE


labelMap = {
    0: "real",
    1: "ai_generated"
}

imgTransform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def predictImage(imgPath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyCNN(numClasses=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    img = Image.open(imgPath).convert("RGB")
    imgTensor = imgTransform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(imgTensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predClass = torch.argmax(probs).item()
    confidence = probs[predClass].item()
    aiScore = probs[1].item()

    result = {
        "predicted_label": labelMap[predClass],
        "confidence_score": round(confidence, 4),
        "ai_generated_score": round(aiScore, 4)
    }

    return result


if __name__ == "__main__":
    result = predictImage("sample.jpg")
    print(result)