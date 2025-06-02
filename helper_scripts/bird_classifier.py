#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class BirdDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class BirdClassifierModel(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifierModel, self).__init__()
        # Use a more powerful pre-trained model (ResNet50 instead of ResNet18)
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class BirdClassifierNode(Node):
    def __init__(self, model_path, class_names):
        super().__init__('bird_classifier_node')
        
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BirdClassifierModel(len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Store class names
        self.class_names = class_names
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers and publishers
        self.image_subscription = self.create_subscription(
            Image,
            '/bird_detector/bird_image',  # Subscribe to detected bird images
            self.image_callback,
            10)
        self.classification_publisher = self.create_publisher(
            String,
            '/bird_classifier/result',
            10)
            
        self.get_logger().info('Bird Classifier Node has been initialized')
        
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to PIL Image for transformation
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Preprocess the image
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Classify the image
            with torch.no_grad():
                output = self.model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Get top prediction
                top_prob, top_class = torch.max(probabilities, 0)
                top3_probs, top3_classes = torch.topk(probabilities, 3)
                self.get_logger().info(f'Top 3 predictions: {top3_classes.cpu().numpy()} with probabilities {top3_probs.cpu().numpy()}')
                predicted_class = self.class_names[top_class.item()]
                confidence = top_prob.item()
                
            # Publish result
            result_msg = String()
            result_msg.data = f"{predicted_class}:{confidence:.4f}"
            self.classification_publisher.publish(result_msg)
            
            self.get_logger().info(f'Bird classified as: {predicted_class} with confidence: {confidence:.4f}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def load_dataset(data_dir):
    """Load images and labels from directory structure with enhanced preprocessing"""
    images = []
    labels = []
    class_names = []
    
    # Iterate through class directories
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, image_file)
                    try:
                        image = PILImage.open(image_path).convert('RGB')
                        
                        # Basic preprocessing - ensure minimum quality
                        # Resize to reasonable dimension if too small
                        if min(image.size) < 224:
                            ratio = 224 / min(image.size)
                            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                            image = image.resize(new_size, PILImage.LANCZOS)
                            
                        images.append(image)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
    
    print(f"Loaded {len(images)} images across {len(class_names)} classes")
    return images, labels, class_names

def train_model(data_dir, model_save_path, batch_size=16, num_epochs=30, learning_rate=0.0001):
    """Train the bird classifier model with enhanced augmentation and training"""
    # Load dataset
    images, labels, class_names = load_dataset(data_dir)
    
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BirdDataset(X_train, y_train, transform=train_transform)
    val_dataset = BirdDataset(X_val, y_val, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Two-stage training strategy:
    # 1. First train with frozen backbone
    # 2. Then fine-tune all layers
    
    model = BirdClassifierModel(len(class_names))
    model.to(device)
    
    # Stage 1: Train only the fully-connected layers (freeze backbone)
    print("Stage 1: Training only fully-connected layers...")
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Only train the final fully connected layer
    for param in model.model.fc.parameters():
        param.requires_grad = True
    
    # Define loss function and optimizer for stage 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=learning_rate)
    
    # Learning rate scheduler - removed verbose parameter
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=3)
    
    # Stage 1: Train for 5 epochs with frozen backbone
    initial_epochs = 5
    best_val_acc = 0.0
    for epoch in range(initial_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        print(f"Stage 1 - Epoch {epoch+1}/{initial_epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    
    # Stage 2: Fine-tune all layers
    print("\nStage 2: Fine-tuning all layers...")
    # Unfreeze all layers
    for param in model.model.parameters():
        param.requires_grad = True
    
    # Use smaller learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate/10, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=3)
    
    # Training loop for stage 2
    patience = 10
    patience_counter = 0
    
    for epoch in range(initial_epochs, num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        print(f"Stage 2 - Epoch {epoch+1}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    # Test Time Augmentation for final evaluation
    print("\nPerforming final evaluation with Test Time Augmentation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()

    # Define TTA transformations
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ]

    # Create a list to store all predictions for each image
    all_image_preds = []
    all_labels = []

    # First, collect all validation images and labels
    for images, labels in val_loader:
        all_labels.extend(labels.numpy())

    # Process each validation image with TTA
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(zip(X_val, y_val)):
            # Apply TTA to this single image
            augmented_predictions = []
            
            # Apply different transformations
            for transform in tta_transforms:
                augmented_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
                output = model(augmented_image)
                _, pred = torch.max(output, 1)
                augmented_predictions.append(pred.item())
            
            # Take majority vote from augmentations
            from collections import Counter
            final_pred = Counter(augmented_predictions).most_common(1)[0][0]
            
            # Update accuracy
            if final_pred == label:
                correct += 1
            total += 1
            
            # Print progress occasionally
            if i % 100 == 0:
                print(f"Processed {i}/{len(X_val)} validation images")
                
    final_accuracy = correct / total
    print(f"Final TTA accuracy: {final_accuracy:.4f}")

    # Save class names alongside the model
    class_names_file = os.path.splitext(model_save_path)[0] + '_classes.txt'
    with open(class_names_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    return class_names

def test_single_image(model_path, image_path):
    """Test the model on a single image"""
    # Load class names
    class_names_file = os.path.splitext(model_path)[0] + '_classes.txt'
    if os.path.exists(class_names_file):
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        print(f"Class names file not found: {class_names_file}")
        return
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BirdClassifierModel(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Load and preprocess the image
    try:
        image = PILImage.open(image_path).convert('RGB')
        
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_classes = torch.topk(probabilities, 3)
            
            print("\n--- Classification Results ---")
            print(f"Image: {image_path}")
            for i in range(3):
                class_idx = top_classes[i].item()
                prob = top_probs[i].item()
                print(f"#{i+1}: {class_names[class_idx]} - {prob:.4f}")
                
            # Optional: Display the image with prediction
            try:
                plt.figure(figsize=(6, 6))
                plt.imshow(np.array(image))
                plt.title(f"Prediction: {class_names[top_classes[0].item()]}")
                plt.axis('off')
                plt.savefig("prediction_result.png")
                print(f"Result image saved as prediction_result.png")
            except Exception as e:
                print(f"Couldn't display image: {e}")
            
    except Exception as e:
        print(f"Error processing image: {e}")
    
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Bird classifier training and inference')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--data_dir', type=str, default='data/birds', help='Directory with training data')
    parser.add_argument('--model_path', type=str, default='models/bird_classifier.pth', help='Path to save/load model')
    parser.add_argument('--inference', action='store_true', help='Run inference using ROS')
    parser.add_argument('--test_image', type=str, help='Test a single image')
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    if args.train:
        print("Starting model training...")
        class_names = train_model(args.data_dir, args.model_path)
    
    if args.test_image:
        test_single_image(args.model_path, args.test_image)
    
    if args.inference:
        # Initialize ROS node
        rclpy.init()
        
        # Load class names
        class_names_file = os.path.splitext(args.model_path)[0] + '_classes.txt'
        if os.path.exists(class_names_file):
            with open(class_names_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            print(f"Class names file not found: {class_names_file}")
            return
        
        # Create and run the ROS node
        node = BirdClassifierNode(args.model_path, class_names)
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
            
if __name__ == '__main__':
    main()