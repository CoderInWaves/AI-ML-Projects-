import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

dataset_dir = '/home/happy/Desktop/Dementia/Final_project/dataset_mri'

folders = {
    'train/normal': [],
    'train/dementia': [],
    'val/normal': [],
    'val/dementia': []
}

print("="*60)
print("MRI DATASET CHECK")
print("="*60)

for folder_name in folders.keys():
    folder_path = os.path.join(dataset_dir, folder_name)
    if os.path.exists(folder_path):
        images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        folders[folder_name] = [os.path.join(folder_path, img) for img in images]
        print(f"\n{folder_name}: {len(images)} images")
        
        if len(images) == 0:
            print(f"ERROR: {folder_name} has ZERO images!")
    else:
        print(f"\nERROR: {folder_path} does not exist!")

print("\n" + "="*60)
print("IMAGE ANALYSIS")
print("="*60)

all_images_to_check = []

for folder_name, image_paths in folders.items():
    if len(image_paths) > 0:
        sample_size = min(5, len(image_paths))
        samples = random.sample(image_paths, sample_size)
        
        print(f"\n{folder_name} - Checking {sample_size} random images:")
        
        for img_path in samples:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            print(f"  {os.path.basename(img_path)}")
            print(f"    Shape: {img_array.shape}")
            print(f"    Min pixel: {img_array.min()}")
            print(f"    Max pixel: {img_array.max()}")
            
            if img_array.max() == 0:
                print(f"    WARNING: All black image!")
            elif img_array.min() == 255 and img_array.max() == 255:
                print(f"    WARNING: All white image!")
            
            all_images_to_check.append((img_path, folder_name))

print("\n" + "="*60)
print("DISPLAYING SAMPLE IMAGES")
print("="*60)

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample MRI Images from Dataset', fontsize=16)

for idx, (img_path, folder_name) in enumerate(all_images_to_check[:20]):
    row = idx // 5
    col = idx % 5
    
    img = Image.open(img_path)
    axes[row, col].imshow(img)
    axes[row, col].set_title(folder_name.split('/')[1], fontsize=8)
    axes[row, col].axis('off')

for idx in range(len(all_images_to_check), 20):
    row = idx // 5
    col = idx % 5
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('/home/happy/Desktop/Dementia/Final_project/mri_dataset_samples.png', dpi=150, bbox_inches='tight')
print(f"\nSample images saved to: mri_dataset_samples.png")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
total_train = len(folders['train/normal']) + len(folders['train/dementia'])
total_val = len(folders['val/normal']) + len(folders['val/dementia'])
print(f"Total training images: {total_train}")
print(f"Total validation images: {total_val}")
print(f"Total images: {total_train + total_val}")
print("="*60)
