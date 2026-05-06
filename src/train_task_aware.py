import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
import os
import glob
import sys

# --- PATH CONFIGURATION ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

try:
    from model_task_aware import TaskAwareGenerator, DiseaseClassifier
    print("Success: Model architectures loaded.")
except ImportError:
    print("Error: model_task_aware.py not found.")
    sys.exit(1)

# --- RESUME SETTINGS ---
RESUME_TRAINING = True 
START_EPOCH = 18  # 15 mukammal hain, ab 16 se 20 ka final phase

def train_jointly():
    # Dummy WandB to bypass Windows block
    class DummyWandb:
        def init(self, *args, **kwargs): return self
        def log(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass

    wandb = DummyWandb()
    print("--- PHASE 3: FINAL FINE-TUNING (Epoch 16-20) ---")
    print("WandB bypassed. LR reduced for crystal clarity.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    generator = TaskAwareGenerator().to(device)
    classifier = DiseaseClassifier(num_classes=9).to(device)

    # --- LOAD SAVED WEIGHTS (Epoch 15) ---
    if RESUME_TRAINING:
        gen_path = f"models/task_aware_gen_epoch_{START_EPOCH}.pth"
        cls_path = f"models/task_aware_classifier_epoch_{START_EPOCH}.pth"
        
        if os.path.exists(gen_path) and os.path.exists(cls_path):
            generator.load_state_dict(torch.load(gen_path, map_location=device))
            classifier.load_state_dict(torch.load(cls_path, map_location=device))
            print(f"Resuming from Epoch {START_EPOCH} weights.")
        else:
            print(f"Error: Epoch {START_EPOCH} checkpoints not found!")
            return

    pixel_criterion = nn.MSELoss()
    task_criterion = nn.CrossEntropyLoss()

    # Learning rate kam kar di hai taake grid pattern (checkerboard) settle ho jaye
    optimizer_G = optim.Adam(generator.parameters(), lr=0.00005) 
    optimizer_C = optim.Adam(classifier.parameters(), lr=0.00002)

    dataset_path = os.path.abspath(os.path.join(current_script_dir, "..", "data", "raw", "SLIF-Brinjal", "Phase_I_Dataset"))
    
    class TaskAwareDataset(Dataset):
        def __init__(self, root_dir):
            self.image_paths = []
            self.labels = []
            self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                for img_path in glob.glob(os.path.join(cls_folder, "*.jpg")):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])
        def __len__(self): return len(self.image_paths)
        def __getitem__(self, idx):
            hr_image = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            lr_t = T.Compose([T.Resize((128, 128)), T.ToTensor()])(hr_image)
            hr_t = T.Compose([T.Resize((512, 512)), T.ToTensor()])(hr_image)
            return lr_t, hr_t, label

    dataset = TaskAwareDataset(dataset_path)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"--- STARTING FINAL PUSH: EPOCH {START_EPOCH + 1} TO 20 ---")
    
    for epoch in range(START_EPOCH, 20):
        generator.train()
        classifier.train()
        for batch_idx, (lr_imgs, hr_imgs, labels) in enumerate(train_loader):
            lr_imgs, hr_imgs, labels = lr_imgs.to(device), hr_imgs.to(device), labels.to(device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            sr_imgs = generator(lr_imgs)
            loss_pixel = pixel_criterion(sr_imgs, hr_imgs)
            outputs_sr = classifier(sr_imgs)
            loss_task = task_criterion(outputs_sr, labels)
            
            # Weight 0.5 barkarar rakha hai sharpness ke liye
            loss_G = loss_pixel + (0.5 * loss_task) 
            
            loss_G.backward()
            optimizer_G.step()

            # --- Train Classifier ---
            optimizer_C.zero_grad()
            outputs_hr = classifier(hr_imgs)
            loss_C = task_criterion(outputs_hr, labels)
            loss_C.backward()
            optimizer_C.step()

            if batch_idx % 10 == 0:
                # Ab aapko dono losses terminal par nazar ayenge
                print(f"Epoch [{epoch+1}] Batch [{batch_idx}] | G-Loss: {loss_G.item():.4f} | C-Loss: {loss_C.item():.4f}")

        # Save after each epoch
        os.makedirs("models", exist_ok=True)
        torch.save(generator.state_dict(), f"models/task_aware_gen_epoch_{epoch+1}.pth")
        torch.save(classifier.state_dict(), f"models/task_aware_classifier_epoch_{epoch+1}.pth")
        print(f"Successfully saved Epoch {epoch+1}")

if __name__ == "__main__":
    train_jointly()