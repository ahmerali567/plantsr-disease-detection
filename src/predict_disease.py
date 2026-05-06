import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys

# Path setup
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)
from model_task_aware import TaskAwareGenerator, DiseaseClassifier

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    gen = TaskAwareGenerator().to(device)
    gen.load_state_dict(torch.load("models/task_aware_gen_epoch_20.pth"))
    gen.eval()
    
    cls = DiseaseClassifier(num_classes=9).to(device)
    cls.load_state_dict(torch.load("models/task_aware_classifier_epoch_20.pth"))
    cls.eval()

    # 2. Dataset Classes (Inhi names se folders hain aapke paas)
    class_names = ['Cercospora_Leaf_Spot', 'Healthy_Leaf', 'Little_Leaf', 'Mosaic_Virus', 'Small_Leaf', 'Other_Diseases'] # Isay apne folders ke mutabiq check krlein

    # 3. Test on a Random Image
    import glob, random
    test_path = glob.glob("data/raw/SLIF-Brinjal/Phase_I_Dataset/**/*.jpg", recursive=True)
    img_path = random.choice(test_path)
    
    img = Image.open(img_path).convert("RGB")
    img_t = T.Compose([T.Resize((128, 128)), T.ToTensor()])(img).unsqueeze(0).to(device)

    # 4. Process & Predict
    with torch.no_grad():
        sr_img = gen(img_t)
        output = cls(sr_img)
        prob = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        confidence = torch.max(prob).item() * 100

    print(f"\n--- DISEASE DETECTION REPORT ---")
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted Disease: {class_names[pred_idx] if pred_idx < len(class_names) else 'Unknown'}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"--------------------------------\n")

if __name__ == "__main__":
    predict()