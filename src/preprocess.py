import cv2
import os
import glob
from tqdm import tqdm

def create_uav_pairs(raw_folder, processed_folder, scale_factor=4):
    lr_path = os.path.join(processed_folder, 'LR')
    hr_path = os.path.join(processed_folder, 'HR')
    os.makedirs(lr_path, exist_ok=True)
    os.makedirs(hr_path, exist_ok=True)

    # Yeh line har sub-folder ke andar ghus kar images dhoondegi
    extensions = ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg', '*.JPEG']
    # Search for all images in ALL subfolders
    image_files = []
    # Yeh pattern har folder ke har level par search karega
    for ext in ('**/*.jpg', '**/*.JPG', '**/*.png', '**/*.PNG', '**/*.jpeg'):
        image_files.extend(glob.glob(os.path.join(raw_folder, ext), recursive=True))

    print(f"Checking in: {os.path.abspath(raw_folder)}")
    print(f"Found {len(image_files)} images.")

    if len(image_files) == 0:
        print("ERROR: Check the folder path")
        return

    for img_path in tqdm(image_files, desc="Processing Images"):
        img = cv2.imread(img_path)
        if img is None: continue

        # Unique name 
        folder_name = os.path.basename(os.path.dirname(img_path))
        filename = f"{folder_name}_{os.path.basename(img_path)}"
        
        # Save HR (512x512)
        hr_img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(hr_path, filename), hr_img)

        # Save LR (128x128) - UAV Simulation
        lr_img = cv2.resize(hr_img, (128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_path, filename), lr_img)

if __name__ == "__main__":

    RAW_DATA = "data/raw" 
    PROCESSED_DATA = "data/processed"
    
    create_uav_pairs(RAW_DATA, PROCESSED_DATA)