import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import random
import glob

# Path setup
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_script_dir)
from model_task_aware import TaskAwareGenerator, DiseaseClassifier

def generate_thesis_panel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating panel on: {device}")

    # 1. Load Models (Epoch 20)
    gen = TaskAwareGenerator().to(device)
    gen.load_state_dict(torch.load("models/task_aware_gen_epoch_20.pth", map_location=device))
    gen.eval()
    
    cls = DiseaseClassifier(num_classes=9).to(device)
    cls.load_state_dict(torch.load("models/task_aware_classifier_epoch_20.pth", map_location=device))
    cls.eval()

    # Dataset folders
    class_names = ['Cercospora_Leaf_Spot', 'Healthy_Leaf', 'Little_Leaf', 'Mosaic_Virus', 'Small_Leaf', 'Other_Diseases']

    # 2. Dataset Paths
    base_data_path = os.path.join(current_script_dir, "..", "data", "raw", "SLIF-Brinjal", "Phase_I_Dataset")
    
    # 3. Select a Random Sick Leaf Image
    sick_leaves = glob.glob(f"{base_data_path}/**/*.jpg", recursive=True)
            
    if not sick_leaves:
        print("Error: Dataset images nahi milien! Path check karein.")
        return
        
    img_path = random.choice(sick_leaves)
    print(f"Processing image: {os.path.basename(img_path)}")
    
    # Load original High-Res
    hr_img_orig = Image.open(img_path).convert("RGB")
    
    # 4. Generate Low-Res and Super-Resolved
    transform_lr = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    lr_tensor = transform_lr(hr_img_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = gen(lr_tensor)
        output = cls(sr_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(prob, dim=1).item()
        confidence = torch.max(prob).item() * 100

    predicted_disease = class_names[pred_idx] if pred_idx < len(class_names) else 'Unknown Disease'

    # 5. Conversion for Visualization
    lr_img_for_view = T.ToPILImage()(lr_tensor.squeeze(0).cpu()).resize((512, 512), Image.NEAREST)
    sr_img_for_view = T.ToPILImage()(sr_tensor.squeeze(0).cpu()).resize((512, 512))
    hr_img_for_view = hr_img_orig.resize((512, 512))

    # 6. Create Comparison Panel
    w, h = 512, 512
    margin = 50
    final_img = Image.new('RGB', (w*3 + margin*4, h + 300), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)

    # Use default font
    try:
        font_title = ImageFont.truetype("arial.ttf", 35)
        font_pred = ImageFont.truetype("arial.ttf", 45)
    except:
        font_title = ImageFont.load_default()
        font_pred = ImageFont.load_default()

    # Draw images
    final_img.paste(lr_img_for_view, (margin, margin))
    final_img.paste(sr_img_for_view, (w + margin*2, margin))
    final_img.paste(hr_img_for_view, (w*2 + margin*3, margin))

    # Labels
    draw.text((margin, h + margin + 10), "Low-Res Input (128x128)", fill=(0,0,0), font=font_title)
    draw.text((w + margin*2, h + margin + 10), "Super-Resolved (512x512)", fill=(0,0,0), font=font_title)
    draw.text((w*2 + margin*3, h + margin + 10), "Original HR (512x512)", fill=(0,0,0), font=font_title)

    # Result Box
    draw.rectangle([w + margin*2, h + margin + 60, w*2 + margin*2, h + margin + 220], outline=(255,0,0), width=5)
    draw.text((w + margin*2 + 20, h + margin + 70), "AI PREDICTION:", fill=(255,0,0), font=font_title)
    draw.text((w + margin*2 + 20, h + margin + 120), f"Disease: {predicted_disease}", fill=(0,0,0), font=font_pred)
    draw.text((w + margin*2 + 20, h + margin + 170), f"Conf: {confidence:.2f}%", fill=(0,128,0), font=font_pred)

    # Save
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs("results", exist_ok=True)
    save_path = f"results/thesis_panel_{base_name}.png"
    final_img.save(save_path)
    print(f"Panel saved successfully: {save_path}")

if __name__ == "__main__":
    generate_thesis_panel()