import copy
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import cv2

# Hyper Parameters_
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 30
model_name = ""
backbone_model = None
all_model_names = [
    "vgg16",
    "densenet121",
    "efficientnet_b0",
    "resnet18",
    "resnet34",
]
all_backbone_models = [
    models.vgg16(pretrained=True),
    models.densenet121(pretrained=True),
    models.efficientnet_b0(pretrained=True),
    models.resnet18(pretrained=True),
    models.resnet34(pretrained=True),
]


class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]


class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img


transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    analyzer = ModelAnalyzer(model, device)
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        val_metrics = evaluate_model(model, val_loader, device)# Update analyzer
        analyzer.update_metrics(
            epoch_loss,
            val_metrics[0],  # Using kappa as validation loss
            train_metrics[1],  # accuracy
            val_metrics[1],  # accuracy
            train_metrics[0],  # kappa
            val_metrics[0]  # kappa
        )
       
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    # Plot metrics at the end of training
    analyzer.plot_metrics()

    return model


def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall


class MyModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.backbone = backbone_model
        state_dict = torch.load(f"./pretrained_DR_resize/pretrained/{model_name}.pth", map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.fc = nn.Identity()  # Remove the original classification layer

        # Unfreeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class MyDualModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super().__init__()

        backbone = backbone_model
        state_dict = torch.load(f"./pretrained_DR_resize/pretrained/{model_name}.pth", map_location='cpu')
        backbone.load_state_dict(state_dict, strict=False)

        if model_name != "vgg16":
            if model_name == "efficientnet_b0" or model_name == "densenet121":
                backbone.classifier = nn.Identity()
            else:
                backbone.fc = nn.Identity()
        else:
            backbone = backbone.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Here the two backbones will have the same structure but unshared weights
        self.backbone1 = copy.deepcopy(backbone)
        self.backbone2 = copy.deepcopy(backbone)

        # Unfreeze all layers
        for param in self.backbone1.parameters():
            param.requires_grad = True
        for param in self.backbone2.parameters():
            param.requires_grad = True

        if model_name == "efficientnet_b0":
            self.fc = nn.Sequential(
                nn.Linear(1280 * 2, 512),  # 1280*2 because we concatenate features from both images
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes)
            )
        elif model_name == "densenet121":
            self.fc = nn.Sequential(
                nn.Linear(1024 * 2, 512),  # 1024*2 because we're concatenating two feature vectors
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(512 * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes)
            )

    def forward(self, images):
        image1, image2 = images

        x1 = self.backbone1(image1)
        x2 = self.backbone2(image2)

        if model_name == "vgg16":
            x1 = self.avgpool(x1)
            x2 = self.avgpool(x2)
            x1 = torch.flatten(x1, 1)
            x2 = torch.flatten(x2, 1)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x


class ModelAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_kappas = []
        self.val_kappas = []
        
    def update_metrics(self, epoch_train_loss, epoch_val_loss, train_accuracy, val_accuracy, train_kappa, val_kappa):
        self.train_losses.append(epoch_train_loss)
        self.val_losses.append(epoch_val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.train_kappas.append(train_kappa)
        self.val_kappas.append(val_kappa)
    
    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        os.makedirs(f"plots/{model_name}")
        
        plt.tight_layout()
        plt.savefig(f"plots/{model_name}/model_metrics.png")
        plt.close()
        
        # Plot Kappa scores
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.train_kappas, 'b-', label='Training Kappa')
        plt.plot(epochs, self.val_kappas, 'r-', label='Validation Kappa')
        plt.title('Model Kappa Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Kappa Score')
        plt.legend()
        plt.savefig(f"plots/{model_name}/kappa_scores.png")
        plt.close()

class DualModelWrapper(torch.nn.Module):
    """Wrapper for analyzing one backbone of a dual model"""
    def __init__(self, model, backbone_idx):
        super().__init__()
        self.full_model = model
        self.backbone_idx = backbone_idx
        self.backbone = model.backbone1 if backbone_idx == 1 else model.backbone2
        
    def forward(self, x):
        # Create a dummy tensor for the other backbone
        dummy = torch.zeros_like(x)
        if self.backbone_idx == 1:
            features = self.full_model([x, dummy])
        else:
            features = self.full_model([dummy, x])
        return features

def get_target_layer(model, backbone_idx=None):
    """Get the target layer for GradCAM based on model architecture"""
    if isinstance(model, MyDualModel):
        backbone = model.backbone1 if backbone_idx == 1 else model.backbone2
    else:
        backbone = model.backbone if hasattr(model, 'backbone') else model
        
    # Handle different architectures
    if model_name == "efficientnet_b0":
        # For EfficientNet, use the last conv layer
        return backbone.features[-1]
    elif model_name == "vgg16":
        conv_layers = [module for name, module in backbone.named_modules() if isinstance(module, nn.Conv2d)]
        return conv_layers[-1]
    elif model_name == "densenet121":
        # For DenseNet, use the last dense block
        return backbone.features.denseblock4
    else:
        # For ResNet architectures
        return backbone.layer4[-1]
    
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[0][target_class] = 1
        
        # Backward pass
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.data.cpu().numpy()[0]
        features = self.features.data.cpu().numpy()[0]
        
        # Calculate weights based on global average pooling
        weights = np.mean(gradients, axis=(1, 2))
        
        # Generate weighted feature map
        cam = np.zeros(features.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * features[i, :, :]
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        
        return cam, target_class

def apply_gradcam(model, image_path, save_path, transform, backbone_idx=None):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    
    # Get the target layer and prepare model
    if isinstance(model, MyDualModel):
        if backbone_idx not in [1, 2]:
            raise ValueError("backbone_idx must be 1 or 2 for dual model")
        
        # Create a wrapper for the specific backbone
        wrapped_model = DualModelWrapper(model, backbone_idx)
        target_layer = get_target_layer(model, backbone_idx)
        model_for_cam = wrapped_model
    else:
        target_layer = get_target_layer(model)
        model_for_cam = model
    
    # Initialize GradCAM
    grad_cam = GradCAM(model_for_cam, target_layer)
    
    # Generate heatmap
    heatmap, predicted_class = grad_cam.generate_cam(input_tensor)
    
    # Convert original image to numpy array
    original_image = np.array(image.resize((224, 224)))
    
    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend original image with heatmap
    superimposed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
    
    # Save visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(heatmap, cmap='jet')
    ax2.set_title('GradCAM Heatmap')
    ax2.axis('off')
    
    ax3.imshow(superimposed)
    ax3.set_title(f'Overlay (Class {predicted_class})')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return predicted_class

def analyze_dataset(model, dataset, save_dir, transform, num_samples=5):
    """
    Analyze multiple images from the dataset using GradCAM
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    model = model.to(device)
    
    for i in range(min(num_samples, len(dataset))):
        if dataset.mode == 'single':
            image_path = dataset.data[i]['img_path']
            dr_level = dataset.data[i].get('dr_level', 'unknown')
            save_path = os.path.join(save_dir, f'gradcam_sample_{i}_dr{dr_level}.png')
            
            predicted_class = apply_gradcam(model, image_path, save_path, transform)
            print(f'Processed image {i+1}/{num_samples}: DR Level {dr_level}, Predicted {predicted_class}')
        else:
            # Handle dual image case
            image_path1 = dataset.data[i]['img_path1']
            image_path2 = dataset.data[i]['img_path2']
            dr_level = dataset.data[i].get('dr_level', 'unknown')
            
            save_path1 = os.path.join(save_dir, f'gradcam_sample_{i}_image1_dr{dr_level}.png')
            save_path2 = os.path.join(save_dir, f'gradcam_sample_{i}_image2_dr{dr_level}.png')
            
            predicted_class1 = apply_gradcam(model, image_path1, save_path1, transform, backbone_idx=1)
            predicted_class2 = apply_gradcam(model, image_path2, save_path2, transform, backbone_idx=2)
            print(f'Processed dual images {i+1}/{num_samples}: DR Level {dr_level}, ' 
                  f'Predicted {predicted_class1}/{predicted_class2}')
            
if __name__ == '__main__':
    for index, name in enumerate(all_model_names):
        model_name = name
        backbone_model = all_backbone_models[index]

        if model_name == "densenet121":
            batch_size = 16

        # Choose between 'single image' and 'dual images' pipeline
        # This will affect the model definition, dataset pipeline, training and evaluation

        # mode = 'single'  # forward single image to the model each time
        mode = 'dual'  # forward two images of the same eye to the model and fuse the features

        assert mode in ('single', 'dual')

        # Define the model
        if mode == 'single':
            model = MyModel()
        else:
            model = MyDualModel()

        print(model, '\n')
        print('Pipeline Mode:', mode)

        # Create datasets
        train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
        val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
        test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Define the weighted CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

        # Use GPU device is possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device:', device)

        # Move class weights to the device
        model = model.to(device)

        # Optimizer and Learning rate scheduler
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train and evaluate the model with the training and validation set
        model = train_model(
            model, train_loader, val_loader, device, criterion, optimizer,
            lr_scheduler=lr_scheduler, num_epochs=num_epochs,
            checkpoint_path=f"./results/part_b_double_{model_name}.pth"
        )

        # Load the pretrained checkpoint
        state_dict = torch.load(f"./results/part_b_double_{model_name}.pth", map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

        # Make predictions on testing set and save the prediction results
        evaluate_model(model, test_loader, device, test_only=True, prediction_path=f"./results/test_prediction_b_double_{model_name}.csv")


        # Analyze images
        analyze_dataset(model, val_dataset, f"./gradcam_results/{model_name}", transform_test, num_samples=10)