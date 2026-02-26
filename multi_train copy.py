"""Thie file contains the main code for training the relevant networks.

Available models for training include:
- CLS model (with/without imagenet pretraining)
|   - grconvnet
|   - alexnet
- Grasp model (with/without imagenet pretraining)
|   - alexnet

Comment or uncomment certain lines of code for swapping between
training CLS model and Grasping model.

E.g. Uncomment the lines with NO SPACE between '#' and the codes: 
"Training for Grasping"
# Loss fn for CLS training
#loss = nn.CrossEntropyLoss()
# Loss fn for Grasping
loss = nn.MSELoss()

----->

# Loss fn for CLS training
loss = nn.CrossEntropyLoss()
# Loss fn for Grasping
#loss = nn.MSELoss()

"""
import copy
import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.models import alexnet

from tqdm import tqdm
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from utils.paths import Path
from utils.parameters import Params
from data_processing.data_loader_v2 import DataLoader
from utils.utils import epoch_logger, log_writer, get_correct_cls_preds_from_map, get_acc
from utils.grasp_utils import get_correct_grasp_preds_from_map
from training_utils.evaluation import get_cls_acc, get_grasp_acc
from training_utils.loss import MapLoss, DistillationLoss


params = Params() 
paths = Path()
SEED = params.SEED
# Create <trained-models> directory
paths.create_model_path()
# Create directory for training logs
paths.create_log_path()
# Create subdirectory in <logs> for current model
paths.create_model_log_path()

# Set common seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# Load model
model =  Multi_AlexnetMap_v3().to(params.DEVICE)


# Create DataLoader class
data_loader = DataLoader(params.TRAIN_PATH, params.BATCH_SIZE, params.TRAIN_VAL_SPLIT, seed=SEED)
# Get number of training/validation steps
n_train, n_val = data_loader.get_train_val()

# Training utils
optim = Adam(model.parameters(), lr=params.LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, 0.5)

# Early stopping setup
best_val_loss = float('inf')
patience = 15  # Number of epochs to wait for improvement
patience_counter = 0
best_model_path = os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_{SEED}_best.pth")

for epoch in tqdm(range(1, params.EPOCHS + 1)):
    if epoch == 10:
        model.unfreeze_depth_backbone()

    train_history = []
    c_train_history = []
    val_history = []
    c_val_history = []
    train_total = 1
    train_correct = 1
    val_total = 1
    val_correct = 1
    
    image_data = enumerate(zip(data_loader.load_grasp_batch(), data_loader.load_batch()))
    values = (0,0)
    values = (0,0)
    for step, ((img_grp, map_grp, label_grp), (img_cls, map_cls, label_cls)) in image_data:
        optim.zero_grad()
        output_cls = model(img_cls, is_grasp=False)
        loss_cls = MapLoss(output_cls, map_cls)
        output_grp = model(img_grp, is_grasp=True)
        loss_grp = MapLoss(output_grp, map_grp)
        # Loss fn for CLS/Grasp training
        loss = (params.LOSS_WEIGHT)*loss_grp + (2 - params.LOSS_WEIGHT) * loss_cls
        # Distillation loss (experimental)
        #distill_loss = DistillationLoss(img, model, pretrained_alexnet, model_s_type='alexnetMap', model_t_type='alexnet')
        #loss = loss + distill_loss * params.DISTILL_ALPHA
        if step < n_train:
            loss.backward()
            optim.step()

            # Write loss to log file -- 'logs/<model_name>/<model_name>_log.txt'
            # log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=True)
            train_history.append(loss_grp)
            c_train_history.append(loss_cls)
            # Dummie prediction stats
            correct, total = 0, 1
            train_correct += correct
            train_total += total
        else:
            # log_writer(params.MODEL_NAME, epoch, step, loss.item(), train=False)
            val_history.append(loss_grp)
            c_val_history.append(loss_cls)
            # Dummie prediction stats
            correct, total = 0, 1
            val_correct += correct
            val_total += total
                
    # Get testing accuracy stats (CLS / Grasp)
    if (epoch % 10 == 1):
        model.eval()
        c_train_acc, c_train_loss = get_cls_acc(model, include_depth=True, seed=SEED, dataset=params.TRAIN_PATH, truncation=None)
        train_acc, train_loss = get_grasp_acc(model, include_depth=True, seed=SEED, dataset=params.TRAIN_PATH, truncation=None)
        c_test_acc, c_test_loss = get_cls_acc(model, include_depth=True, seed=SEED, dataset=params.TEST_PATH, truncation=None)
        test_acc, test_loss = get_grasp_acc(model, include_depth=True, seed=SEED, dataset=params.TRAIN_PATH, truncation=None)
        scheduler.step()
        
        # Early stopping check - use combined validation loss
        current_val_loss = (sum(val_history) / len(val_history) if val_history else 0) + \
                          (sum(c_val_history) / len(c_val_history) if c_val_history else 0)
        
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save best model to disk
            torch.save(model.state_dict(), best_model_path)
            print(f"\nEpoch {epoch}: Validation loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"\nEpoch {epoch}: No improvement. Patience: {patience_counter}/{patience}")
        
        # Check if we should stop
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            # Restore best model from disk
            model.load_state_dict(torch.load(best_model_path))
            break
        
        # Experimental
        #params.DISTILL_ALPHA /= 2
        
        model.train()

    # Get training and validation accuracies
    val_acc = train_acc # get_acc(val_correct, val_total)
    c_val_acc = c_train_acc
    # Write epoch loss stats to log file
    epoch_logger(params.MODEL_NAME, epoch, train_history, val_history, test_loss, train_acc, val_acc, test_acc, c_train_history, c_val_history, c_test_loss, c_train_acc, c_val_acc, c_test_acc)
    # Save checkpoint model -- 'trained-models/<model_name>/<model_name>_epoch<epoch>.pth'
    #torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_{SEED}_epoch{epoch}.pth"))

# Save final/best epoch model
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path))
torch.save(model.state_dict(), os.path.join(params.MODEL_LOG_PATH, f"{params.MODEL_NAME}_{SEED}_final.pth"))