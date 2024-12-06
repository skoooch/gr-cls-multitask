import enum
import torch
from sklearn.model_selection import train_test_split
from model import Multi_AlexnetMap_Width
from torch.optim import Adam
from parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from tqdm import tqdm
from utils.parameters import Params
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import h5py
import numpy as np
params = Params()
num_epochs = 10
model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
model =  Multi_AlexnetMap_Width(weights_path).to('cuda')
batch_size = 4
data = torch.load("goodale/rectangle_dataset/indices.pt", weights_only=True)[:, 0, :]
perm = torch.randperm(data.shape[0])
idx = perm[:1000]
data = data[idx]
y = (data[:, 0] + 30).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
optim = Adam(model.parameters(), lr=params.LR)
loss_fn = torch.nn.MSELoss()
running_loss = 0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/goodale_sim_{}'.format(timestamp))
output_dir = '/scratch/expires-2024-Nov-26/'
os.makedirs(output_dir, exist_ok=True)
filename = f'rect_data_angled3.hdf5'
filepath = os.path.join(output_dir, filename)
h5_file = h5py.File(filepath, 'r')
image_data = h5_file.get("data")
best_vloss = 1_000_000.
for epoch in range(num_epochs):
    running_loss = 0.
    last_loss = 0.
    batch_total = 0
    model.train(True)
    for i, data in tqdm(enumerate(training_loader), total = len(training_loader)):
        batch_total = i + 1
        inputs, labels = data

        # Zero your gradients for every batch!
        optim.zero_grad()
        img_set = np.ndarray((batch_size,4, 224,224,4))
        for j in range(inputs.shape[0]):     
            img_set[j] = image_data[inputs[j, 0], inputs[j, 1], inputs[j, 2]]
        # shape = (batch_size, 4, 224,224, 4)
        random_indices_dim1 = torch.randint(0, img_set.shape[1], (batch_size,))
        # Make predictions for this batch
        img = torch.Tensor(img_set[torch.arange(batch_size), random_indices_dim1]).to("cuda")
        #shape = (batch_size, 224,224,4)
        img = torch.where(img == 2, 0.1, img)
        img = torch.where(img == 1, -0.1, img)
        outputs = model(img.permute(0, 3, 1, 2))
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs[:, 0], labels.to("cuda"))
        loss.backward()

        # Adjust learning weights
        optim.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(training_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    avg_loss = last_loss
    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            img_set = np.ndarray((batch_size,4, 224,224,4))
            for j in range(vinputs.shape[0]):     
                img_set[j] = image_data[vinputs[j, 0], vinputs[j, 1], vinputs[j, 2]]
            # shape = (batch_size, 4, 224,224, 4)
            random_indices_dim1 = torch.randint(0, img_set.shape[1], (batch_size,))
            # Make predictions for this batch
            img = torch.Tensor(img_set[torch.arange(batch_size), random_indices_dim1]).to("cuda")
            #shape = (batch_size, 224,224,4)
            img = torch.where(img == 2, 0.1, img)
            img = torch.where(img == 1, -0.1, img)
            voutputs = model(img.permute(0, 3, 1, 2))
            
            # Compute the loss and its gradients
            vloss = loss_fn(voutputs[:, 0], vlabels.to("cuda"))
            running_vloss += vloss
    avg_vloss = running_vloss / (len(validation_loader))
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'goodale/trained_models/model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)
model_path = 'goodale/trained_models/model_{}_final'.format(timestamp, epoch)
torch.save(model.state_dict(), model_path)