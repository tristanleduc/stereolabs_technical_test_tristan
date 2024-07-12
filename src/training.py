import torch
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import CityscapesSkyDataset, ADE20KSkyDataset
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

image_train_dir_cityscapes = os.path.join(os.getcwd(), 'dataset', 'leftImg8bit', 'train')
mask_train_dir_cityscapes = os.path.join(os.getcwd(), 'dataset', 'gtFine', 'train')
image_val_dir_cityscapes = os.path.join(os.getcwd(), 'dataset', 'leftImg8bit', 'val')
mask_val_dir_cityscapes = os.path.join(os.getcwd(), 'dataset', 'gtFine', 'val')

image_dir_ade20K = os.path.join(os.getcwd(), '..', 'dataset', 'ADE20K_2021_17_01', 'images', 'ADE')

train_dataset_cityscapes = CityscapesSkyDataset(image_dir=image_train_dir_cityscapes, mask_dir=mask_train_dir_cityscapes, transform=transform, target_transform=target_transform)
val_dataset_cityscapes = CityscapesSkyDataset(image_dir=image_val_dir_cityscapes, mask_dir=mask_val_dir_cityscapes, transform=transform, target_transform=target_transform)

train_dataset_ade20k = ADE20KSkyDataset(image_dir=image_dir_ade20K, index_file='index_ade20k.pkl', transform=transform, target_transform=target_transform, split='train')
val_dataset_ade20k = ADE20KSkyDataset(image_dir=image_dir_ade20K, index_file='index_ade20k.pkl', transform=transform, target_transform=target_transform, split='val')

random_indices = random.sample(range(0, len(train_dataset_ade20k)), 4000) 
train_dataset_ade20k = torch.utils.data.Subset(train_dataset_ade20k, random_indices)

# concatenating the two datasets 
train_dataset = train_dataset_ade20k

train_loader = DataLoader(train_dataset, batch_size=13, shuffle=True)
val_loader_cityscapes = DataLoader(val_dataset_cityscapes, batch_size=17, shuffle=False)
val_loader_ade20k = DataLoader(val_dataset_ade20k, batch_size=13, shuffle=False)

# loading the model weights saved in pth 
model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
#model.load_state_dict(torch.load('../models/sky_segmentation_model_15_epochs.pth'))

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device {device}')
model = model.to(device)

# Training parameters
num_epochs = 30
initial_lr = 0.0008

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

writer = SummaryWriter(log_dir='runs/sky_segmentation_6')

criterion = torch.nn.BCEWithLogitsLoss()
criterion.to(device)

def save_checkpoint(epoch, model, optimizer, loss, filename="checkpoint.pth.tar"):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)

if __name__ == "__main__":
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + len(progress_bar))

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train_avg', avg_loss, epoch)

        if (epoch + 1) % 2 == 0:
            print("Validating...")
            model.eval()
            val_loss_cityscapes = 0.0
            val_loss_ade20k = 0.0
            with torch.no_grad():
                for images, masks in val_loader_cityscapes:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                    val_loss_cityscapes += loss.item()

                for images, masks in val_loader_ade20k:
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                    val_loss_ade20k += loss.item()
                    writer.add_images('images', images, epoch)
                    writer.add_images('masks', masks, epoch)
                    writer.add_images('outputs', outputs, epoch)
            print("Validation done.")

            val_loss_cityscapes /= len(val_loader_cityscapes)
            val_loss_ade20k /= len(val_loader_ade20k)
            writer.add_scalar('Loss/val_cityscapes', val_loss_cityscapes, epoch)
            writer.add_scalar('Loss/val_ade20k', val_loss_ade20k, epoch)

            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss (Cityscapes): {val_loss_cityscapes}, Validation Loss (ADE20K): {val_loss_ade20k}')

            # Save checkpoint if validation loss has improved
            if val_loss_ade20k < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss_ade20k:.4f}. Saving checkpoint.")
                save_checkpoint(epoch + 1, model, optimizer, val_loss_ade20k, filename=f"checkpoint_best.pth.tar")
                best_val_loss = val_loss_ade20k

            # Stepping the scheduler with the validation loss
            scheduler.step(val_loss_ade20k)

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch + 1, model, optimizer, avg_loss, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")

    torch.save(model.state_dict(), 'sky_segmentation_model_25_epochs.pth')
    writer.close()