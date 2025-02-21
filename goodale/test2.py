import enum
import torch
from sklearn.model_selection import train_test_split
from model import Multi_AlexnetMap_Width
from torch.optim import Adam
from parameters import Params
from multi_task_models.grcn_multi_alex import Multi_AlexnetMap_v3
from tqdm import tqdm
#from utils.parameters import Params
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

output_dir = '/scratch/expires-2025-Feb-06/'
os.makedirs(output_dir, exist_ok=True)
filename = f'rect_data_angled_test.hdf5'
filepath = os.path.join(output_dir, filename)
h5_file = h5py.File(filepath, 'r')
image_data = h5_file.get("data")
data_points = []
model.eval()
i = 0
total_loss = 0
for width in range(30, 120,5):
    for height in range(30, 120,10):
        i += 1
        img = torch.Tensor(image_data[width-30, (height-30)//2, 0, 0]).to("cuda")

        img = torch.where(img == 2, 0.1, img)
        img = torch.where(img == 1, -0.1, img)
        outputs = model(img.permute(2, 0, 1)[None, :, :, :])
        total_loss += abs(width - outputs[0][0].detach().cpu())
        print(width)
        print(outputs[0][0].detach().cpu())
print(total_loss/i)
