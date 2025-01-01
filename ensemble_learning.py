import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class EnsembleModel:
    def __init__(self, models, device):
        self.models = models
        self.device = device
        for model in self.models:
            model.eval()
            model.to(device)

    def weighted_average_prediction(self, loader, weights=None):
        """
        Make predictions using weighted average of model outputs
        """
        if weights is None:
            weights = [1/len(self.models)] * len(self.models)
        
        all_preds = []
        true_labels = []
        
        with torch.no_grad():
            for data in tqdm(loader, desc="Predicting"):
                if len(data) == 2:  # Training/validation mode
                    images, labels = data
                    true_labels.extend(labels.numpy())
                else:  # Test mode
                    images = data
                
                if not isinstance(images, list):
                    images = images.to(self.device)
                else:
                    images = [x.to(self.device) for x in images]

                # Get predictions from each model
                ensemble_outputs = torch.zeros((images[0].size(0) if isinstance(images, list) else images.size(0), 5)).to(self.device)
                
                for i, model in enumerate(self.models):
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    ensemble_outputs += weights[i] * probabilities

                predictions = torch.argmax(ensemble_outputs, dim=1)
                all_preds.extend(predictions.cpu().numpy())
        
        return all_preds, true_labels if true_labels else None

    def max_voting_prediction(self, loader):
        """
        Make predictions using max voting (majority voting)
        """
        all_preds = []
        true_labels = []
        
        with torch.no_grad():
            for data in tqdm(loader, desc="Predicting"):
                if len(data) == 2:
                    images, labels = data
                    true_labels.extend(labels.numpy())
                else:
                    images = data
                
                if not isinstance(images, list):
                    images = images.to(self.device)
                else:
                    images = [x.to(self.device) for x in images]

                # Get predictions from each model
                model_predictions = []
                for model in self.models:
                    outputs = model(images)
                    predictions = torch.argmax(outputs, dim=1)
                    model_predictions.append(predictions.cpu().numpy())

                # Use majority voting
                ensemble_predictions = np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=5).argmax(),
                    axis=0,
                    arr=np.array(model_predictions)
                )
                all_preds.extend(ensemble_predictions)
        
        return all_preds, true_labels if true_labels else None

    def get_features(self, loader):
        """
        Extract features from all models for stacking
        """
        all_features = []
        true_labels = []
        
        with torch.no_grad():
            for data in tqdm(loader, desc="Extracting features"):
                if len(data) == 2:
                    images, labels = data
                    true_labels.extend(labels.numpy())
                else:
                    images = data
                
                if not isinstance(images, list):
                    images = images.to(self.device)
                else:
                    images = [x.to(self.device) for x in images]

                batch_features = []
                for model in self.models:
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    batch_features.append(probabilities.cpu().numpy())
                
                # Concatenate features from all models
                batch_features = np.concatenate(batch_features, axis=1)
                all_features.extend(batch_features)
        
        return np.array(all_features), np.array(true_labels) if true_labels else None

class StackingEnsemble:
    def __init__(self, base_ensemble, meta_learner=None):
        self.base_ensemble = base_ensemble
        self.meta_learner = meta_learner if meta_learner else LogisticRegression(max_iter=1000)

    def fit(self, train_loader, val_loader):
        """
        Train the stacking ensemble
        """
        # Get features and labels from training data
        train_features, train_labels = self.base_ensemble.get_features(train_loader)
        
        # Train meta-learner
        self.meta_learner.fit(train_features, train_labels)
        
        # Evaluate on validation data if provided
        if val_loader:
            val_features, val_labels = self.base_ensemble.get_features(val_loader)
            val_predictions = self.meta_learner.predict(val_features)
            return val_predictions, val_labels
        
        return None, None

    def predict(self, test_loader):
        """
        Make predictions using the stacking ensemble
        """
        # Get features from test data
        test_features, _ = self.base_ensemble.get_features(test_loader)
        
        # Use meta-learner for final predictions
        predictions = self.meta_learner.predict(test_features)
        return predictions

def load_ensemble_models(model_paths, mode='dual'):
    """
    Load the pretrained models for ensemble
    """
    models = []
    for path in model_paths:
        if mode == 'single':
            model = MyModel()
        else:
            model = MyDualModel()
        
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        models.append(model)
    
    return models

def evaluate_ensemble(predictions, labels):
    """
    Evaluate ensemble predictions
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score
    
    accuracy = accuracy_score(labels, predictions)
    kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    return kappa, accuracy, precision, recall

# Example usage:
def run_ensemble_evaluation(model_paths, val_loader, test_loader, device):
    """
    Run and evaluate different ensemble methods
    """
    # Load models
    models = load_ensemble_models(model_paths)
    ensemble = EnsembleModel(models, device)
    
    # 1. Weighted Average
    weights = [0.3, 0.3, 0.4]  # Can be tuned based on individual model performance
    pred_weighted, labels = ensemble.weighted_average_prediction(val_loader, weights)
    metrics_weighted = evaluate_ensemble(pred_weighted, labels)
    print("\nWeighted Average Ensemble:")
    print(f"Kappa: {metrics_weighted[0]:.4f}, Accuracy: {metrics_weighted[1]:.4f}")
    print(f"Precision: {metrics_weighted[2]:.4f}, Recall: {metrics_weighted[3]:.4f}")
    
    # 2. Max Voting
    pred_voting, labels = ensemble.max_voting_prediction(val_loader)
    metrics_voting = evaluate_ensemble(pred_voting, labels)
    print("\nMax Voting Ensemble:")
    print(f"Kappa: {metrics_voting[0]:.4f}, Accuracy: {metrics_voting[1]:.4f}")
    print(f"Precision: {metrics_voting[2]:.4f}, Recall: {metrics_voting[3]:.4f}")
    
    # 3. Stacking
    stacking = StackingEnsemble(ensemble)
    pred_stacking, labels = stacking.fit(val_loader, val_loader)
    metrics_stacking = evaluate_ensemble(pred_stacking, labels)
    print("\nStacking Ensemble:")
    print(f"Kappa: {metrics_stacking[0]:.4f}, Accuracy: {metrics_stacking[1]:.4f}")
    print(f"Precision: {metrics_stacking[2]:.4f}, Recall: {metrics_stacking[3]:.4f}")
    
    # Make predictions on test set using best method
    # Choose the best method based on validation performance
    metrics = [metrics_weighted, metrics_voting, metrics_stacking]
    best_method_idx = np.argmax([m[0] for m in metrics])  # Using kappa score
    
    if best_method_idx == 0:
        test_predictions, _ = ensemble.weighted_average_prediction(test_loader, weights)
        method_name = "Weighted Average"
    elif best_method_idx == 1:
        test_predictions, _ = ensemble.max_voting_prediction(test_loader)
        method_name = "Max Voting"
    else:
        test_predictions = stacking.predict(test_loader)
        method_name = "Stacking"
    
    return test_predictions, method_name

# Save predictions to CSV
def save_predictions(predictions, image_ids, output_path):
    import pandas as pd
    df = pd.DataFrame({
        'ID': image_ids,
        'TARGET': predictions
    })
    df.to_csv(output_path, index=False)