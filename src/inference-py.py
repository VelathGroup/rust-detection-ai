import torch
from torchvision import transforms
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Models definition
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 2, stride=2), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, 2, stride=2), torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

def load_models(models_dir="models"):
    """Load all models required for the inference pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load YOLO Steel/Rust Detector
    yolo_path = os.path.join(models_dir, 'yolo_best.pt')
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
    
    yolo_model = YOLO(yolo_path)
    print("✅ YOLO model loaded successfully")

    # Load UNet Edge Detector
    unet_path = os.path.join(models_dir, 'unet_edge_detector.pth')
    if not os.path.exists(unet_path):
        raise FileNotFoundError(f"UNet model not found at {unet_path}")
    
    unet_model = SimpleUNet()
    unet_model.load_state_dict(torch.load(unet_path, map_location=device))
    unet_model.to(device)
    unet_model.eval()
    print("✅ UNet edge detector loaded successfully")

    # Load Rust Level Classifier
    from torchvision import models
    classifier_path = os.path.join(models_dir, 'rust_level_classifier.pth')
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier model not found at {classifier_path}")
    
    rust_classifier = models.mobilenet_v3_small(pretrained=False)
    rust_classifier.classifier[3] = torch.nn.Linear(1024, 2)
    rust_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    rust_classifier.to(device)
    rust_classifier.eval()
    print("✅ Rust level classifier loaded successfully")
    
    return yolo_model, unet_model, rust_classifier, device

def robot_perception_inference(image_path, yolo_model, unet_model, rust_classifier, device, save_output=True, output_dir=None):
    """
    Run the complete inference pipeline on an image.
    
    Args:
        image_path (str): Path to the input image
        yolo_model: YOLO object detection model
        unet_model: UNet edge detection model
        rust_classifier: MobileNetV3 rust level classifier
        device: Device to run inference on (CPU/GPU)
        save_output (bool): Whether to save the output visualization
        output_dir (str): Directory to save output visualization
        
    Returns:
        dict: Results containing detection information and decisions
    """
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    print(f"Processing image: {image_path}")
    
    # Prepare transforms
    transform_unet = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    transform_rust = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Open Image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # 1. YOLO Detection
    yolo_result = yolo_model.predict(image_path)[0]
    boxes = yolo_result.boxes.data.cpu().numpy()

    steel_detected = False
    rust_detected = False
    
    # Parse YOLO results
    # Class indices: 0=steel_rust, 1=steel_clean, 2=nonsteel_rust, 3=nonsteel_clean
    class_names = {0: 'steel_rust', 1: 'steel_clean', 2: 'nonsteel_rust', 3: 'nonsteel_clean'}
    detected_objects = []

    for box in boxes:
        cls = int(box[5])
        conf = box[4]
        class_name = class_names.get(cls, f"class_{cls}")
        detected_objects.append({"class": class_name, "confidence": float(conf)})
        
        if cls in [0, 1]:  # Steel classes
            steel_detected = True
        if cls in [0, 2]:  # Rust classes
            rust_detected = True

    # 2. UNet Edge Detection
    unet_input = transform_unet(image).unsqueeze(0).to(device)
    with torch.no_grad():
        edge_map = unet_model(unet_input).squeeze().cpu().numpy()

    # 3. Rust Level Classification
    rust_level = None
    rust_confidence = None
    if rust_detected:
        rust_input = transform_rust(image).unsqueeze(0).to(device)
        with torch.no_grad():
            out = rust_classifier(rust_input)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(out, dim=1).item()
            rust_level = 'SA2.5' if pred == 0 else 'SA3.0'
            rust_confidence = float(probs[0][pred].cpu().numpy())

    # 4. Blasting Decision
    decision = None
    if steel_detected:
        print("✅ Steel Surface Detected.")
        print("✅ Edge Map Generated.")
        if rust_detected:
            print(f"⚡ Rust Detected - Rust Level: {rust_level} (confidence: {rust_confidence:.2f})")
            if rust_level == 'SA2.5':
                decision = "Light Blasting Recommended"
                print(f"💨 Decision: {decision}.")
            elif rust_level == 'SA3.0':
                decision = "Heavy Blasting Required"
                print(f"💥 Decision: {decision}.")
        else:
            decision = "No Blasting Needed"
            print(f"✅ Surface is Clean. {decision}.")
    else:
        print("❌ No Steel Surface Detected.")
        decision = "No Action Required - Not Steel"

    # 5. Visualization
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,3,1)
    plt.imshow(image_np)
    plt.title("Original Image")
    
    # Plot YOLO detections
    plt.subplot(1,3,2)
    yolo_img = yolo_result.plot()
    plt.imshow(yolo_img)
    plt.title("Object Detection")

    plt.subplot(1,3,3)
    plt.imshow(edge_map, cmap='gray')
    plt.title("Edge Map Prediction")
    
    plt.suptitle(f"Decision: {decision}", fontsize=16)
    plt.tight_layout()
    
    # Save output if requested
    if save_output:
        if output_dir is None:
            output_dir = "results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"result_{filename}")
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")
    
    plt.show()
    
    # Return structured results
    results = {
        "steel_detected": steel_detected,
        "rust_detected": rust_detected,
        "rust_level": rust_level,
        "rust_confidence": rust_confidence,
        "detected_objects": detected_objects,
        "decision": decision
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Robot Perception Inference Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--save_output', action='store_true', help='Save output visualization')
    
    args = parser.parse_args()
    
    # Load models
    yolo_model, unet_model, rust_classifier, device = load_models(args.models_dir)
    
    # Run inference
    results = robot_perception_inference(
        args.image, 
        yolo_model, 
        unet_model, 
        rust_classifier, 
        device,
        save_output=args.save_output,
        output_dir=args.output_dir
    )
    
    # For automated systems, you could return the result as JSON
    import json
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
