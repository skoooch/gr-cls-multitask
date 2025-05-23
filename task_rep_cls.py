import torch
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v3 import DataLoader

params = Params()

class LinearProbe(nn.Module):
    def __init__(self, input_dim=5, output_dim=4):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class FeatureExtractor:
    def __init__(self, model, layer_name='features.10', kernel_subset=[1, 20, 32, 42, 45]):
        self.model = model
        self.kernel_subset = kernel_subset
        self.features = {}
        
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook_fn)
                break
                
    def hook_fn(self, module, input, output):
        self.features['layer'] = output.detach()
        
    def extract_features(self, x):
        with torch.no_grad():
            _ = self.model(x)
            features = self.features['layer']
            features = features[:, self.kernel_subset, :, :]
            features = torch.mean(features, dim=[2, 3])
        return features

def collect_all_data(dataloader, extractor):
    all_features = []
    all_targets = []
    
    for batch in dataloader.load_task_rep_batch():
        img_batch, colour_r, colour_g, colour_b, colour_a, _, _ = batch
        
        features = extractor.extract_features(img_batch)
        targets = torch.stack([colour_r, colour_g, colour_b, colour_a], dim=1)
        
        all_features.append(features.cpu())
        all_targets.append(targets.cpu())
    
    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

def train_with_cv(features, targets, device, epochs=10, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    os.makedirs('./task_rep_weights', exist_ok=True)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f'Fold {fold+1}/{n_splits}')
        
        train_features = features[train_idx].to(device)
        train_targets = targets[train_idx].to(device)
        val_features = features[val_idx].to(device)
        val_targets = targets[val_idx].to(device)
        
        probe = LinearProbe().to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            probe.train()
            optimizer.zero_grad()
            outputs = probe(train_features)
            loss = criterion(outputs, train_targets)
            loss.backward()
            optimizer.step()
            
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(val_features)
                val_loss = criterion(val_outputs, val_targets)
                
                r2_scores = []
                for i in range(4):
                    r2 = r2_score(val_targets[:, i].cpu().numpy(), val_outputs[:, i].cpu().numpy())
                    r2_scores.append(r2)
                
                print(f'Epoch {epoch+1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, R2: {np.mean(r2_scores):.4f}')
        
        fold_results.append(np.mean(r2_scores))
        torch.save(probe.state_dict(), f'./task_rep_weights/probe_fold_{fold+1}.pth')
    
    print(f'CV Results: Mean R2: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')
    
    probe = LinearProbe().to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    features = features.to(device)
    targets = targets.to(device)
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(features)
                r2_scores = []
                for i in range(4):
                    r2 = r2_score(targets[:, i].cpu().numpy(), val_outputs[:, i].cpu().numpy())
                    r2_scores.append(r2)
                print(f'Final Training Epoch {epoch+1}: R2: {np.mean(r2_scores):.4f}')
    
    torch.save(probe.state_dict(), './task_rep_weights/probe_final.pth')
    return probe

def evaluate_test(extractor, device):
    dataloader_test = DataLoader(params.TEST_PATH, params.BATCH_SIZE, task_rep_path=params.TEST_TASK_REP_PATH, device=device)
    
    probe = LinearProbe().to(device)
    probe.load_state_dict(torch.load('./task_rep_weights/probe_final.pth'))
    probe.eval()
    
    test_features = []
    test_targets = []
    
    for batch in dataloader_test.load_task_rep_batch():
        img_batch, colour_r, colour_g, colour_b, colour_a, _, _ = batch
        
        features = extractor.extract_features(img_batch)
        targets = torch.stack([colour_r, colour_g, colour_b, colour_a], dim=1)
        
        test_features.append(features.cpu())
        test_targets.append(targets.cpu())
    
    test_features = torch.cat(test_features, dim=0).to(device)
    test_targets = torch.cat(test_targets, dim=0).to(device)
    
    with torch.no_grad():
        test_outputs = probe(test_features)
        criterion = nn.MSELoss()
        test_loss = criterion(test_outputs, test_targets)
        
        r2_scores = []
        mse_scores = []
        for i in range(4):
            r2 = r2_score(test_targets[:, i].cpu().numpy(), test_outputs[:, i].cpu().numpy())
            mse = mean_squared_error(test_targets[:, i].cpu().numpy(), test_outputs[:, i].cpu().numpy())
            r2_scores.append(r2)
            mse_scores.append(mse)
    
    print(f'Test Results: Loss: {test_loss.item():.4f}, R2: {np.mean(r2_scores):.4f}, MSE: {np.mean(mse_scores):.4f}')
    print(f'Individual R2 - R: {r2_scores[0]:.4f}, G: {r2_scores[1]:.4f}, B: {r2_scores[2]:.4f}, A: {r2_scores[3]:.4f}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Multi_AlexnetMap_v3().to(device)
    model_name = params.MODEL_NAME
    weights_dir = params.MODEL_PATH
    weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    extractor = FeatureExtractor(model)
    
    dataloader_train = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, task_rep_path=params.TRAIN_TASK_REP_PATH, device=device)
    
    train_features, train_targets = collect_all_data(dataloader_train, extractor)
    
    train_with_cv(train_features, train_targets, device, epochs=10)
    evaluate_test(extractor, device)