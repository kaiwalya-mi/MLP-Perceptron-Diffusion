import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def save_image(tensor, filename, size=(256, 256)):
    """Save a tensor as an image"""
    img_array = tensor.detach().cpu().numpy()
    img_array = img_array.reshape(size[0], size[1], 3)
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)

def create_animation(input_dir, output_file, fps=10):
    """Create an animation from a directory of images"""
    import imageio
    
    # Get all image files
    files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    
    # Create animation
    with imageio.get_writer(output_file, mode='I', fps=fps) as writer:
        for file in files:
            image = imageio.imread(file)
            writer.append_data(image)
            
    print(f'Animation saved to {output_file}')

def load_model(model, checkpoint_path):
    """Load a model from a checkpoint"""
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def generate_image_from_model(model, width=256, height=256):
    """Generate an image from a trained model"""
    # Generate coordinates
    x_coords = np.linspace(0, 1, width)
    y_coords = np.linspace(0, 1, height)
    
    coords = []
    for y in y_coords:
        for x in x_coords:
            coords.append([x, y])
    
    coords_tensor = torch.FloatTensor(np.array(coords))
    
    # Generate colors
    with torch.no_grad():
        colors = model(coords_tensor)
    
    # Reshape to image
    img_array = colors.numpy().reshape(height, width, 3)
    
    return img_array
