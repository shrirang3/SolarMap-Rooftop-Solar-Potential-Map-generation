import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from optimized_unet_model import OptimizedUNet
import argparse

def load_model(model_path):
    """Load the trained UNet model."""
    model = OptimizedUNet(n_channels=3, n_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Move the model to the appropriate device
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess the input image for the model."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Apply the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(image).unsqueeze(0)

def predict_mask(model, image_tensor):
    """Generate prediction mask."""
    with torch.no_grad():
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Get model prediction
        output = model(image_tensor)
        prob_mask = torch.sigmoid(output)
        
        # Convert to binary mask
        binary_mask = (prob_mask > 0.7).float()
        
        return prob_mask.cpu().numpy(), binary_mask.cpu().numpy()

def visualize_results(original_image, probability_mask, binary_mask, save_path=None):
    """Visualize and optionally save the segmentation results."""
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot probability mask
    prob_plot = axes[1].imshow(probability_mask[0, 0], cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Probability Mask')
    axes[1].axis('off')
    plt.colorbar(prob_plot, ax=axes[1])
    
    # Plot binary mask
    axes[2].imshow(binary_mask[0, 0], cmap='binary')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict segmentation mask for a single image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='new_best_unet_model.pth', help='Path to trained model')
    parser.add_argument('--output_path', type=str, default='prediction_result.png', help='Path to save visualization')
    parser.add_argument('--save_mask', type=str, default='predicted_mask.png', help='Path to save binary mask')
    
    args = parser.parse_args()
    
    try:
        # Load the model
        model = load_model(args.model_path)
        
        # Preprocess the image
        image = Image.open(args.image_path).convert('RGB')
        image_tensor = preprocess_image(args.image_path)
        
        # Generate prediction
        prob_mask, binary_mask = predict_mask(model, image_tensor)
        
        # Visualize and save results
        visualize_results(image, prob_mask, binary_mask, args.output_path)
        
        # Save the binary mask as an image
        mask_image = Image.fromarray((binary_mask[0, 0] * 255).astype(np.uint8))
        mask_image.save(args.save_mask)
        
        print(f"Results saved to {args.output_path}")
        print(f"Binary mask saved to {args.save_mask}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()



##########################################################################################################################################################################

# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# import numpy as np
# import os
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from optimized_unet_model import OptimizedUNet

# class Predictor:
#     def __init__(self, model_path, output_dir='predictions'):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = OptimizedUNet(n_channels=3, n_classes=1).to(self.device)
        
#         # Load the trained model
#         checkpoint = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.eval()
        
#         # Create output directory if it doesn't exist
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Define image transformation
#         self.transform = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                               std=[0.229, 0.224, 0.225])
#         ])

#     def preprocess_image(self, image_path):
#         """Load and preprocess a single image."""
#         image = Image.open(image_path).convert('RGB')
#         original_size = image.size
#         image = self.transform(image)
#         return image.unsqueeze(0).to(self.device), original_size

#     def postprocess_prediction(self, prediction, original_size):
#         """Convert prediction to binary mask and resize to original image size."""
#         prediction = torch.sigmoid(prediction)
#         prediction = (prediction > 0.5).float()
#         prediction = prediction.squeeze().cpu().numpy()
        
#         # Convert to PIL Image and resize to original size
#         mask = Image.fromarray((prediction * 255).astype(np.uint8))
#         mask = mask.resize(original_size, Image.NEAREST)
#         return mask

#     def overlay_prediction(self, original_image, prediction_mask):
#         """Create an overlay of the prediction on the original image."""
#         # Convert prediction mask to RGBA
#         overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
#         prediction_rgba = prediction_mask.convert('RGBA')
        
#         # Create red transparent overlay
#         red_overlay = Image.new('RGBA', original_image.size, (255, 0, 0, 128))
#         overlay.paste(red_overlay, mask=prediction_mask)
        
#         # Combine original image with overlay
#         original_rgba = original_image.convert('RGBA')
#         return Image.alpha_composite(original_rgba, overlay)

#     def predict_single_image(self, image_path):
#         """Predict mask for a single image and save results."""
#         # Load and preprocess image
#         image_tensor, original_size = self.preprocess_image(image_path)
        
#         # Make prediction
#         with torch.no_grad():
#             prediction = self.model(image_tensor)
        
#         # Postprocess prediction
#         prediction_mask = self.postprocess_prediction(prediction, original_size)
        
#         # Create visualization
#         original_image = Image.open(image_path).convert('RGB')
#         overlay_image = self.overlay_prediction(original_image, prediction_mask)
        
#         # Save results
#         base_name = os.path.splitext(os.path.basename(image_path))[0]
#         prediction_mask.save(os.path.join(self.output_dir, f'{base_name}_mask.png'))
#         overlay_image.save(os.path.join(self.output_dir, f'{base_name}_overlay.png'))
        
#         return prediction_mask, overlay_image

#     def predict_batch(self, input_dir):
#         """Predict masks for all images in a directory."""
#         image_files = [f for f in os.listdir(input_dir) 
#                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
#         for image_file in tqdm(image_files, desc="Processing images"):
#             image_path = os.path.join(input_dir, image_file)
#             try:
#                 self.predict_single_image(image_path)
#             except Exception as e:
#                 print(f"Error processing {image_file}: {str(e)}")

# def main():
#     parser = argparse.ArgumentParser(description='Predict segmentation masks using trained U-Net model')
#     parser.add_argument('--model_path', type=str, default='new_best_unet_model.pth',
#                       help='Path to the trained model checkpoint')
#     parser.add_argument('--input', type=str, required=True,
#                       help='Path to input image or directory')
#     parser.add_argument('--output_dir', type=str, default='predictions',
#                       help='Directory to save predictions')
    
#     args = parser.parse_args()
    
#     # Initialize predictor
#     predictor = Predictor(args.model_path, args.output_dir)
    
#     # Process input
#     if os.path.isfile(args.input):
#         # Single image prediction
#         predictor.predict_single_image(args.input)
#         print(f"Prediction saved to {args.output_dir}")
#     elif os.path.isdir(args.input):
#         # Batch prediction
#         predictor.predict_batch(args.input)
#         print(f"All predictions saved to {args.output_dir}")
#     else:
#         print("Invalid input path")

# if __name__ == "__main__":
#     main()




