from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Brain Tumor Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates") 

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False
for layer in model.features[-3:]:
    for param in layer.parameters():
        param.requires_grad = True
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).item()

    prediction = "Tumor" if pred >= 0.5 else "No Tumor"
    confidence = float(pred)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": confidence
        }
    )