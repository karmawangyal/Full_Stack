
import torch
import torchvision.transforms as transforms
def denormalize(image, mean, std):
    denormalized_image = image * std[:, None, None] + mean[:, None, None]
    return denormalized_image


def reverse_to_tensor(image, mean, std):
    image = denormalize(image, mean, std)
    image = image * 255.0
    image = image.byte()
    return image
 

def denormalize_bbox(bbox, image_width, image_height):
    x_min = bbox[0] * image_width  
    y_min = bbox[1] * image_height
    x_max = bbox[2] * image_width
    y_max = bbox[3] * image_height
    return [x_min, y_min, x_max, y_max]

# Transformation for input image
def transform(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    image_tensor = transform_pipeline(image)
    image_tensor = image_tensor.unsqueeze(0)  # Unsqueeze to add a batch dimension
    return image_tensor

