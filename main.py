import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, output_size=3):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# Load and preprocess image
def load_image(image_path, size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img) / 255.0
    return img_array

# Generate coordinate grid
def generate_coordinates(width, height):
    x_coords = np.linspace(0, 1, width)
    y_coords = np.linspace(0, 1, height)
    
    coords = []
    for y in y_coords:
        for x in x_coords:
            coords.append([x, y])
    
    return np.array(coords)

# Main training function
def train_model(image_path, epochs=5000, lr=0.001, save_interval=100):
    # Load image
    img_array = load_image(image_path)
    height, width, _ = img_array.shape
    
    # Generate coordinates
    coords = generate_coordinates(width, height)
    
    # Prepare target colors
    target_colors = img_array.reshape(-1, 3)
    
    # Convert to PyTorch tensors
    coords_tensor = torch.FloatTensor(coords)
    colors_tensor = torch.FloatTensor(target_colors)
    
    # Initialize model and optimizer
    model = MLPModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Training loop
    losses = []
    for epoch in tqdm(range(epochs)):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(coords_tensor)
        loss = criterion(outputs, colors_tensor)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Save intermediate results
        if epoch % save_interval == 0 or epoch == epochs - 1:
            # Generate image from current model
            with torch.no_grad():
                predicted_colors = model(coords_tensor).numpy()
                predicted_img = predicted_colors.reshape(height, width, 3)
                
                # Save the image
                plt.figure(figsize=(10, 10))
                plt.imshow(predicted_img)
                plt.axis('off')
                plt.savefig(f'outputs/epoch_{epoch}.png', bbox_inches='tight')
                plt.close()
            
            # Save model checkpoint
            torch.save(model.state_dict(), f'outputs/model_epoch_{epoch}.pth')
            
            # Print progress
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/loss_curve.png')
    plt.close()
    
    return model

if __name__ == '__main__':
    # Replace with your image path
    image_path = 'sample_image.jpg'
    
    # Train the model
    model = train_model(image_path, epochs=5000, save_interval=100)
    
    print('Training completed!')
