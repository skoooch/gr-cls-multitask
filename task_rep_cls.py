import sys
import test
import torch
import torch.nn as nn
import os
import numpy as np
import torchvision
from torchvision.utils import save_image
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from utils.parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from data_processing.data_loader_v3 import DataLoader
import matplotlib.pyplot as plt
params = Params()

class LinearProbe(nn.Module):
    def __init__(self, input_dim=5*169, output_dim=4, softmax=False):
        super(LinearProbe, self).__init__()
        if not softmax:
            self.linear = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim))
        else:
            self.linear = nn.Sequential(nn.Linear(input_dim, input_dim*2), 
                                        nn.Linear(input_dim*2, input_dim),
                                        nn.Linear(input_dim, output_dim), 
                                        nn.Softmax(dim=1))
    def forward(self, x):
        return self.linear(x)
# [4,6,10,12,13]
#[1,20,32,42,45]
class FeatureExtractor:
    def __init__(self, model, layer_name='features.10', kernel_subset=[4,6,10,12,13]):
        self.model = model
        self.layer_name = layer_name
        self.kernel_subset = kernel_subset
        self.features = {}
        if layer_name == "first":
                for name, module in model.named_modules():
                    if name == 'rgb_features.0':
                        module.register_forward_hook(self.hook_fn_r)
                    elif name == 'd_features.0':
                        module.register_forward_hook(self.hook_fn_d)              
        else:
            for name, module in model.named_modules():
                if name == layer_name:
                    module.register_forward_hook(self.hook_fn)
                    break
                
    def hook_fn(self, module, input, output):
        self.features['layer'] = output.detach()
    def hook_fn_r(self, module, input, output):
        self.features['rgb'] = output.detach() 
    def hook_fn_d(self, module, input, output):
        self.features['d'] = output.detach()  
    def extract_features(self, x):
        if self.layer_name != "first":
            with torch.no_grad():
                _ = self.model(x[:,0,:,:,:])
                features = self.features['layer']
                features = features[:, self.kernel_subset, :, :]
                #features = torch.mean(features, dim=[2, 3])
                features = torch.flatten(features, start_dim=1, end_dim=3)
            return features
        else:
            with torch.no_grad():
                _ = self.model(x[:,0,:,:,:])
                r_features = self.features['rgb']
                d_features = self.features['d']
                rgb_subset = torch.tensor([kernel_idx for kernel_idx in self.kernel_subset if kernel_idx < 64], dtype=torch.long)
                d_subset = torch.tensor([kernel_idx-64 for kernel_idx in self.kernel_subset if kernel_idx >= 64], dtype=torch.long)
                r_features = r_features[:, rgb_subset, :, :]
                d_features = d_features[:, d_subset, :,:]
                features = torch.cat((r_features, d_features), dim=1)
                #features = torch.mean(features, dim=[2, 3])
                features = torch.flatten(features, start_dim=1, end_dim=3)
            return features
def get_accuracy(targets, outputs):
    total = 0
    correct = 0
    for i in range(targets.shape[0]):
        total += 1
        correct += torch.max(torch.tensor(targets[i]), 0)[1] == torch.max(torch.tensor(outputs[i]), 0)[1]
    return correct/total

def collect_all_data(dataloader, extractor, target_type="color"):
    all_features = []
    all_targets = []
    for batch in dataloader.load_task_rep_batch():
        img_batch, colour_r, colour_g, colour_b, colour_a, locationx, locationy, angle, comp_angle,sin2,cos2 = batch
        # test = img_batch[1,0,:3,:,:]
        # print(locationx)
        # test[:, locationy.long()[1]//1, locationx.long()[1]//1] = torch.tensor([255,0,0])
        # save_image(test, f"vis/probe/test_sample.png")
        # exit()
        features = extractor.extract_features(img_batch)
        if target_type == "color":
            targets = torch.stack([colour_r, colour_g, colour_b, colour_a], dim=1)
        elif target_type == "angle":
            angles_indices = angle.long() // 90
            targets = torch.nn.functional.one_hot(angles_indices, num_classes=4)
        elif target_type == "comp_angle":
            targets = comp_angle[:, None]
        elif target_type == "sincos":
            targets = torch.stack([sin2, cos2], dim=1)
        elif target_type == "location":
            targets = torch.stack([locationx, locationy], dim=1)
        all_features.append(features.cpu())
        all_targets.append(targets.cpu())
    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

def train_with_cv(features, targets, device, epochs=10, n_splits=5, classify=False, output_dim=1):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    os.makedirs('./task_rep_weights', exist_ok=True)
    
    fold_results = []
    input_dim = 0
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        print(f'Fold {fold+1}/{n_splits}')
        
        train_features = features[train_idx].to(device)
        train_targets = targets[train_idx].to(device)
        val_features = features[val_idx].to(device)
        val_targets = targets[val_idx].to(device)
        input_dim = train_features.shape[1]
        probe = LinearProbe(input_dim=input_dim,output_dim=output_dim,softmax=classify).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
        if not classify: criterion = nn.MSELoss()
        else: criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            probe.train()
            optimizer.zero_grad()
            outputs = probe(train_features)
            loss = criterion(outputs, train_targets.float())
            loss.backward()
            optimizer.step()
            
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(val_features)
                val_loss = criterion(val_outputs, val_targets.float())
                r2_scores = []
                if not classify: 
                    for i in range(val_targets.shape[1]):
                        r2 = r2_score(val_targets[:, i].cpu().numpy(), val_outputs[:, i].cpu().numpy())
                        r2_scores.append(r2)
                else:
                    r2_scores.append(get_accuracy(val_targets.cpu().numpy(), val_outputs.cpu().numpy()))
                print(f'Epoch {epoch+1}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, R2: {np.mean(r2_scores):.4f}')
        
        fold_results.append(np.mean(r2_scores))
        torch.save(probe.state_dict(), f'./task_rep_weights/probe_fold_{fold+1}_{target_type}.pth')
    
    print(f'CV Results: Mean R2: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}')

    probe = LinearProbe(input_dim=input_dim,output_dim=output_dim,softmax=classify).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
    if not classify: criterion = nn.MSELoss()
    else: criterion = nn.CrossEntropyLoss()
    
    features = features.to(device)
    targets = targets.to(device)
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(features)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(features)
                r2_scores = []
                if not classify: 
                    for i in range(targets.shape[1]):
                        r2 = r2_score(targets[:, i].cpu().numpy(), val_outputs[:, i].cpu().numpy())
                        r2_scores.append(r2)
                else:
                    r2_scores.append(get_accuracy(targets.cpu().numpy(), val_outputs.cpu().numpy()))
                #print(f'Final Training Epoch {epoch+1}: R2: {np.mean(r2_scores):.4f}')
    
    torch.save(probe.state_dict(), f'./task_rep_weights/probe_final_{target_type}.pth')
    return probe, input_dim

def evaluate_test(extractor, device, classify=False, target_type="color", output_dim=1, input_dim = 5*169):
    dataloader_test = DataLoader(params.TEST_PATH, params.BATCH_SIZE, task_rep_path=params.TEST_TASK_REP_PATH, device=device, seed = 0)
    
    probe = LinearProbe(input_dim=input_dim,output_dim=output_dim,softmax = classify).to(device)
    probe.load_state_dict(torch.load(f'./task_rep_weights/probe_final_{target_type}.pth'))
    probe.eval()
    
    test_features = []
    test_targets = []
    
    for batch in dataloader_test.load_task_rep_batch():
        img_batch, colour_r, colour_g, colour_b, colour_a, locationx, locationy, angle, comp_angle, sin2, cos2 = batch
        
        features = extractor.extract_features(img_batch)
        if target_type == "color": targets = torch.stack([colour_r, colour_g, colour_b, colour_a], dim=1)
        elif target_type == "angle":
            angles_indices = angle // 90
            targets = torch.nn.functional.one_hot(angles_indices, num_classes=4)
        elif target_type == "comp_angle":
            targets = comp_angle[:, None]
        elif target_type == "sincos":
            targets = torch.stack([sin2, cos2], dim=1)
        elif target_type == "location":
            targets = torch.stack([locationx, locationy], dim=1)
        test_features.append(features.cpu())
        test_targets.append(targets.cpu())
    
    test_features = torch.cat(test_features, dim=0).to(device)
    test_targets = torch.cat(test_targets, dim=0).to(device)
    if not classify:
        with torch.no_grad():
            test_outputs = probe(test_features)
            criterion = nn.MSELoss()
            test_loss = criterion(test_outputs, test_targets.float())
            r2_scores = []
            mse_scores = []
            for i in range(test_targets.shape[1]):
                r2 = r2_score(test_targets[:, i].cpu().numpy(), test_outputs[:, i].cpu().numpy())
                mse = mean_squared_error(test_targets[:, i].cpu().numpy(), test_outputs[:, i].cpu().numpy())
                r2_scores.append(r2)
                mse_scores.append(mse)
        
        print(f'Test Results: Loss: {test_loss.item():.4f}, R2: {np.mean(r2_scores):.4f}, MSE: {np.mean(mse_scores):.4f}')
        if target_type == "color": print(f'Individual R2 - R: {r2_scores[0]:.4f}, G: {r2_scores[1]:.4f}, B: {r2_scores[2]:.4f}, A: {r2_scores[3]:.4f}')
        elif target_type == "comp_angle": print(f'Individual R2 - Angle: {r2_scores[0]:.4f}')
        elif target_type == "sincos": print(f'Individual R2 - sin: {r2_scores[0]:.4f}, cos: {r2_scores[1]:.4f}')
        elif target_type == "location": print(f'Individual R2 - x: {r2_scores[0]:.4f}, y: {r2_scores[1]:.4f}')
        return np.mean(r2_scores)
    else:
        print(get_accuracy(test_targets, probe(test_features)))
def run_experiment(model, device, dataloader_train, target_type="color", output_dim=4):
    diff = False
    r2_set = np.zeros((5,2,8))
    if diff: top = torch.tensor(np.load("shap_arrays/sort_shap_indices_diff_depth.npy"), dtype=int)
    else: top = torch.tensor(np.load("shap_arrays/sort_shap_indices_depth.npy"), dtype=int)
    for num_kernel in range(3,11)[0:1]:    
        print(num_kernel)
        for layer_i, layer_name in enumerate(['first', 'features.0', 'features.4', 'features.7', 'features.10']):
            for task_i, task in enumerate(['class', 'grasp']):
                extractor = FeatureExtractor(model, layer_name=layer_name, kernel_subset=top[layer_i, :num_kernel,task_i])
                train_features, train_targets = collect_all_data(dataloader_train, extractor, target_type=target_type)
                _, input_dim = train_with_cv(train_features, train_targets, device, epochs=15, classify=False, output_dim=output_dim)
                r2_array = []
                for iter in range(1):
                    r2_array.append(evaluate_test(extractor, device, classify=False, target_type=target_type, output_dim=output_dim, input_dim=input_dim))
                r2_set[layer_i, task_i, num_kernel-3] = np.mean(r2_array)
        layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']
        plt.figure(figsize=(10, 6))
        plt.plot(layers, r2_set[:, 0, num_kernel-3], marker='o', label='Ventral Stream', color="orange")
        plt.plot(layers, r2_set[:, 1, num_kernel-3], marker='s', label='Dorsal Stream', color="blue")
        plt.xlabel('Layer')
        plt.ylabel('R2 Score')
        plt.title(f'R2 Score by Layer and Task Type ({target_type}) - {num_kernel} Kernel Used')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'vis/probe/r2_scores_{target_type}_{num_kernel}.png')
        plt.close()
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Multi_AlexnetMap_v3().to(device)
    model_name = params.MODEL_NAME
    weights_dir = params.MODEL_PATH
    weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    target_type = sys.argv[1]
    for param in model.parameters():
        param.requires_grad = False
    dataloader_train = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, task_rep_path=params.TRAIN_TASK_REP_PATH, device=device, seed=0)
        
    run_experiment(model, device, dataloader_train, target_type=target_type, output_dim=2)
    extractor = FeatureExtractor(model)
    
    train_features, train_targets = collect_all_data(dataloader_train, extractor, target_type=target_type)
    
    train_with_cv(train_features, train_targets, device, epochs=10, classify=False, output_dim=2)
    evaluate_test(extractor, device, classify=False, target_type=target_type, output_dim=2)