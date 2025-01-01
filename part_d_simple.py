import copy
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from collections import Counter

# Model architectures
class MyDualModel(nn.Module):
    def __init__(self, backbone_model, model_name, num_classes=5, dropout_rate=0.5):
        super().__init__()

        self.model_name = model_name
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

        if self.model_name == "vgg16":
            x1 = self.avgpool(x1)
            x2 = self.avgpool(x2)
            x1 = torch.flatten(x1, 1)
            x2 = torch.flatten(x2, 1)

        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# Dataset and transformations
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

class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='dual', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.test = test
        self.mode = mode
        self.data = self.load_data_dual() if mode == 'dual' else self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.get_item_dual(index) if self.mode == 'dual' else self.get_item(index)

    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)
        df['prefix'] = df['image_id'].str.split('_').str[0]
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]
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

# Transform definitions
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class EnsemblePredictor:
    def __init__(self, models, device):
        self.models = models
        self.device = device

    def get_predictions(self, dataloader, method='weighted_average', weights=None):
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from each model
        with torch.no_grad():
            for model_idx, model in enumerate(self.models):
                print(f"\nProcessing model {model_idx + 1}/{len(self.models)}")
                model_preds = []
                model_probs = []
                
                for batch_idx, data in enumerate(tqdm(dataloader, desc="Getting predictions")):
                    # Debug prints
                    print(f"\nBatch {batch_idx + 1}:")
                    print(f"Data type: {type(data)}")
                    
                    # Handle different data formats
                    if isinstance(data, list):
                        print("Data is a list")
                        images = data
                    else:
                        print(f"Data is {type(data)}")
                        if len(data) == 2:
                            images, _ = data
                        else:
                            images = data
                    
                    print(f"Images type: {type(images)}")
                    
                    # Handle the dual image case
                    if isinstance(images, list) and len(images) == 2:
                        print("Processing dual images")
                        # If images is already a list of two tensors, move them to device
                        if isinstance(images[0], torch.Tensor):
                            images = [img.to(self.device) for img in images]
                        else:
                            # For the case where images is a list containing another list of tensors
                            images = [img.to(self.device) for img in images[0]]
                    else:
                        print("Processing single image")
                        images = images.to(self.device)
                    
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    model_preds.extend(preds.cpu().numpy())
                    model_probs.append(probs.cpu().numpy())
                
                all_predictions.append(model_preds)
                all_probabilities.append(np.concatenate(model_probs))
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        print(f"\nPredictions shape: {all_predictions.shape}")
        print(f"Probabilities shape: {all_probabilities.shape}")
        
        if method == 'max_voting':
            final_predictions = []
            for i in range(len(all_predictions[0])):
                votes = Counter(all_predictions[:, i])
                final_predictions.append(votes.most_common(1)[0][0])
            return np.array(final_predictions)
            
        elif method == 'weighted_average':
            if weights is None:
                weights = np.ones(len(self.models)) / len(self.models)
            weighted_probs = np.average(all_probabilities, axis=0, weights=weights)
            return np.argmax(weighted_probs, axis=1)
            
        elif method == 'bagging':
            final_predictions = []
            for i in range(len(all_predictions[0])):
                bootstrap_preds = np.random.choice(all_predictions[:, i], size=len(self.models))
                votes = Counter(bootstrap_preds)
                final_predictions.append(votes.most_common(1)[0][0])
            return np.array(final_predictions)
        
        else:
            raise ValueError(f"Unknown method: {method}")

def save_predictions(predictions, dataset, save_path):
    image_ids = []
    for i in range(len(predictions)):
        if dataset.mode == 'single':
            image_ids.append(os.path.basename(dataset.data[i]['img_path']))
        else:
            image_ids.extend([
                os.path.basename(dataset.data[i]['img_path1']),
                os.path.basename(dataset.data[i]['img_path2'])
            ])
    
    df = pd.DataFrame({
        'ID': image_ids,
        'TARGET': np.repeat(predictions, 2) if dataset.mode == 'dual' else predictions
    })
    df.to_csv(save_path, index=False)
    print(f'Saved predictions to {save_path}')

def load_model(model_name, model_path, backbone, device):
    model = MyDualModel(backbone, model_name)
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove 'backbone_model' entries if they exist
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('backbone_model')}
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    all_models = []
    model_configs = [
        ("resnet18", "results/part_b_double_resnet18.pth", models.resnet18(pretrained=True)),
        ("resnet34", "results/part_b_double_resnet34.pth", models.resnet34(pretrained=True)),
        ("densenet121", "results/part_b_double_densenet121.pth", models.densenet121(pretrained=True))
    ]

    for model_name, model_path, backbone in model_configs:
        model = load_model(model_name, model_path, backbone, device)
        model.eval()
        all_models.append(model)
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, 'dual')
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, 'dual', test=True)
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize ensemble predictor
    ensemble = EnsemblePredictor(all_models, device)
    
    # Try different ensemble methods
    methods = ['max_voting', 'weighted_average', 'bagging']
    best_method = None
    best_kappa = -1
    
    # Evaluate on validation set to find best method
    print("\nEvaluating ensemble methods on validation set...")
    for method in methods:
        print(f"\nTesting {method}...")
        val_preds = ensemble.get_predictions(val_loader, method=method)
        
        # Get validation labels
        val_labels = []
        for _, labels in val_loader:
            val_labels.extend(labels.numpy())
        
        kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        print(f"{method} - Validation Kappa: {kappa:.4f}")
        
        if kappa > best_kappa:
            best_kappa = kappa
            best_method = method
    
    print(f"\nBest performing method: {best_method} (Validation Kappa: {best_kappa:.4f})")
    
    # Generate predictions for test set using best method
    print("\nGenerating test predictions using best ensemble method...")
    test_preds = ensemble.get_predictions(test_loader, method=best_method)
    
    # Save predictions
    save_predictions(test_preds, test_dataset, f"./results/test_prediction_ensemble_{best_method}.csv")

if __name__ == "__main__":
    main()