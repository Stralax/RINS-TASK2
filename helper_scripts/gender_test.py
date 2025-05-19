import os
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# Load model and feature extractor
model_name = "dima806/fairface_gender_image_detection"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

# Folder with images
folder_path = "faces"

# Gender labels according to the model config or typical binary gender labels
# According to model card, the labels are usually ["Male", "Female"]
labels = model.config.id2label if hasattr(model.config, "id2label") else {0: "Male", 1: "Female"}

def predict_gender(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = probs.argmax()
        pred_label = labels[pred_id]
        confidence = probs[pred_id]
    return pred_label, confidence

# Statistics for evaluation
correct = 0
total = 0

# Iterate over images and print predictions
print("\nPrediction Results:")
print("-" * 80)
print(f"{'Filename':<30} | {'True Label':<10} | {'Prediction':<10} | {'Confidence':<10} | {'Correct?':<10}")
print("-" * 80)

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        path = os.path.join(folder_path, filename)
        gender, conf = predict_gender(path)
        
        # Extract ground truth from filename (_F for female, _M for male)
        true_gender = None
        if "_F" in filename:
            true_gender = "Female"
        elif "_M" in filename:
            true_gender = "Male"
        
        # Check if prediction is correct
        is_correct = "N/A"
        if true_gender is not None:
            is_correct = "✓" if gender == true_gender else "✗"
            total += 1
            if gender == true_gender:
                correct += 1
        
        print(f"{filename:<30} | {true_gender:<10} | {gender:<10} | {conf:.4f}     | {is_correct:<10}")

# Print summary statistics
if total > 0:
    print("-" * 80)
    print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")