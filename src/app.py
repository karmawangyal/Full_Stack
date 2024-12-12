from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException, status
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from io import BytesIO
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from fastapi.staticfiles import StaticFiles
from .helper import reverse_to_tensor, denormalize_bbox
import os

# FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# CNN model definition
class CNN(torch.nn.Module):
    def __init__(self, num_channels=3, num_classes=4):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)

        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(512)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(512, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3_bbox = torch.nn.Linear(512, 4)

        self.fc2_class = torch.nn.Linear(512, 256)
        self.fc3_class = torch.nn.Linear(256, num_classes)

        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)

        bbox_pr = self.fc3_bbox(x)
        x_class = torch.nn.functional.relu(self.fc2_class(x))
        class_pr = self.fc3_class(x_class)

        return bbox_pr, class_pr

# Load the pre-trained model (adjust path accordingly)
model_path = "src/best_model-3.pth"
model = CNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transformation for input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def get_email_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login/")
async def login(request: Request, email: str = Form(...)):
    # Validate email
    if "@" not in email:
        return templates.TemplateResponse("login.html", {"request": request})
    
    # Store email locally
    try:
        with open("emails.txt", "a") as file:  # Append mode
            file.write(email + "\n")
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to save email: {str(e)}"}, status_code=500)
    
    # If email is valid, redirect to image upload page
    return RedirectResponse(url="/visualize-predictions/", status_code=status.HTTP_302_FOUND)

@app.get("/visualize-predictions/")
async def visualize_predictions(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/visualize-predictions/")
async def visualize_prediction(request: Request, file: UploadFile = File(...)):
    try:
        # Define mean and std for normalization (as used in transform)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        # Define the class labels mapping
        class_labels_dict = {
            0: 'Glioma',
            1: 'Meningioma',
            2: 'No Tumor',
            3: 'Pituitary',
        }

        # Read the uploaded image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))

        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Prediction
        with torch.no_grad():
            predictions = model(image_tensor)
            bbox, class_scores = predictions

            # Process bounding boxes
            final_bbox = [
                [b[0].item(), b[1].item(), b[2].item(), b[3].item()] for b in bbox
            ]
            # Process class scores
            final_class_labels = class_scores.argmax(dim=-1).tolist()

            # Map numeric labels to human-readable labels
            final_class_labels = [class_labels_dict[label] for label in final_class_labels]

        # Reverse transformations on the image
        reversed_image_tensor = reverse_to_tensor(image_tensor.squeeze(0), mean, std)
        reversed_image_pil = Image.fromarray(reversed_image_tensor.permute(1, 2, 0).numpy())

        # Denormalize bounding boxes
        final_bbox_denormalized = [
            denormalize_bbox(b, image_width=256, image_height=256) for b in final_bbox
        ]

        # Draw bounding boxes and labels on the reversed image
        draw = ImageDraw.Draw(reversed_image_pil)
        font = ImageFont.load_default()

        for i, b in enumerate(final_bbox_denormalized):
            x_min, y_min, x_max, y_max = b
            label = final_class_labels[i]

            # Draw rectangle and label
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            font_size = 50
            draw.text((x_min, y_min - 10), label, fill="white", font=font)

        # Convert the image to a format suitable for response
        img_io = BytesIO()
        reversed_image_pil.save(img_io, format="JPEG")
        img_io.seek(0)

        # Create a temporary file path to store the image in a valid directory
        result_image_path = "static/images/image_with_bboxes.jpg"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(result_image_path), exist_ok=True)

        # Save the image to the specified path
        reversed_image_pil.save(result_image_path)

        # Return the result with bounding boxes and labels
        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_path": f"/static/images/image_with_bboxes.jpg" 
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Run the application with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
