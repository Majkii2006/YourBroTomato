from flask import Flask
from flask import render_template, redirect, url_for, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

num_classes = 10

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()





app = Flask(__name__)


@app.route('/home' and '/' ,methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    tensor = transform(img).unsqueeze(0).to(device)


    with torch.inference_mode():
        model.to(device)
        output = model(tensor)

        _, predicted = torch.max(output, 1)     #znalezienie najwiekszej wartosci i zwrocenie jej indeksu

    digit = predicted.item()

    return render_template('predict.html', digit=digit)









if __name__ == "__main__":
    app.run(debug=True)
