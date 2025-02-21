import enum
import torch
from sklearn.model_selection import train_test_split
from model import Multi_AlexnetMap_Width
from torch.optim import Adam
from parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from tqdm import tqdm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import h5py
import numpy as np
from scipy.stats import pearsonr
from matplotlib import cm
from matplotlib.ticker import LinearLocator

torch.manual_seed(0)
params = Params()
model_type = "grasp"
num_epochs = 10
model_name = params.MODEL_NAME
weights_dir = params.MODEL_PATH
weights_path = os.path.join(weights_dir, model_name, model_name + '_final.pth')
weights_path = os.path.join(weights_dir, f'alexnetMap_{model_type}.pth')
model =  Multi_AlexnetMap_Width(weights_path, True).to('cuda')
model_weights_path = f"goodale/trained_models/model_{model_type}_move_final.pth"
model.load_state_dict(torch.load(model_weights_path))
batch_size = 1

data = torch.load("goodale/rectangle_dataset/indices.pt", weights_only=True)[:, 0, :]
perm = torch.randperm(data.shape[0])
idx = perm[:15000]
data = data[idx]
# data = data[data[:, 3] == 0]
# data = data[data[:, 2] == 0] 
y = (data[:, 0] + 30).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, shuffle=True)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = torch.utils.data.TensorDataset(X_validation, y_validation)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
optim = Adam(model.parameters(), lr=params.LR)
loss_fn = torch.nn.MSELoss()
running_loss = 0

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/goodale_sim_{}'.format(model_type, timestamp))
output_dir = '/scratch/expires-2025-Feb-15/'
os.makedirs(output_dir, exist_ok=True)
filename = f'rect_data_angled.hdf5'
filepath = os.path.join(output_dir, filename)
h5_file = h5py.File(filepath, 'r')
image_data = h5_file.get("data")
data_points = []
model.eval()
total_loss = 0
total = 0
for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
    total += 1
    batch_total = i + 1
    inputs, labels = data

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
    outputs = model(img.permute(2, 0, 1)[None, :, :, :])
    data_points.append((inputs[0, 0] + 30, inputs[0, 1]*2 + 30, abs(outputs.item() - inputs[0, 0] - 30)))
    if (i + 1) % 200 == 0:
        x_values = np.array([float(x.item() - y.item()) for x, y, z in data_points])
        #y_values = np.array([float(y.item()) for x, y, z in data_points])
        z_values = np.array([float(z.item()) for x, y, z in data_points])
        fig = plt.figure()
        ax = fig.add_subplot()
        correlation_coefficient, p_value = pearsonr(x_values, z_values)
        # Create a scatter plot
        ax.scatter(x_values, z_values, color='blue')
        
        # Annotate the plot with the correlation coefficient and p-value
        plt.text(0.5, 0.9, f'Correlation: {correlation_coefficient:.2f}\nP-value: {p_value:.3f}',
         horizontalalignment='right', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=12, color='red')
        plt.ylabel('Error in Width Prediction (pixels)')
        plt.xlabel('Difference in Width and Height (pixels)')
        plt.title(f'{model_type} model')
        plt.legend()
        plt.grid(True)
        print("here")
        plt.savefig(f"./goodale/loss_difhalfway_{model_type}.png")
        plt.clf()
