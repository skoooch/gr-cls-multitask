from math import atan2, pi, sin, cos

import torchvision
from data_processing.data_loader_v2 import DataLoader
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
import scipy.spatial as spatial
from shapely.geometry import Polygon
from utils.parameters import Params
import torch
import pickle
import imageio
def angle_of_hull_of_mask(mask):
    indices_where = np.nonzero(mask)
    hull = indices_where[spatial.ConvexHull(indices_where).vertices, :]

    dist_mat = spatial.distance_matrix(hull, hull)

    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    if hull[i][0] > hull[j][0]:
        angle = atan2(hull[i][0] - hull[j][0], hull[i][1] - hull[j][1])
    else:
        angle = atan2(hull[j][0] - hull[i][0], hull[j][1] - hull[i][1])
    return angle, hull[i], hull[j]
def center_of_mask(mask):
    indices_where = np.nonzero(mask)
    hull = indices_where[spatial.ConvexHull(indices_where).vertices, :]
    polygon = Polygon(hull)
    centroid = polygon.centroid
    
    # # Save a copy of the mask with the center marked
    # mask_with_center = mask.copy()
    # cy_int, cx_int = int(round(cy)), int(round(cx))
    # if 0 <= cy_int < mask_with_center.shape[0] and 0 <= cx_int < mask_with_center.shape[1]:
        
    #     mask_with_center = np.concatenate([mask_with_center[None,:,:],mask_with_center[None,:,:],mask_with_center[None,:,:]]) * 255
    #     mask_with_center[0, cy_int, cx_int] = 0
        
    #     # Mark center with a different value (e.g., 2)
    #     # Optionally save the mask as an image for visualization
        
    # exit()
    return centroid.x, centroid.y

if __name__ == "__main__":
    params = Params()
    
    splits = {
        'train': params.TRAIN_PATH,
        'test': params.TEST_PATH
    }
    
    for split in splits.keys():
        
        dl = DataLoader(
            path=splits[split],
            batch_size=16,
            return_mask=True,
            return_angle=True, seed=0
        )
        
        fa_data = {
            'colour_r': [],
            'colour_g': [],
            'colour_b': [],
            'colour_a': [],
            'location_x': [],
            'location_y': [],
            'angle':[],
            'comp_angle':[],
            'sin2':[],
            'cos2':[],
            'img': []
        }
        
        # Preprocess CLS
        with tqdm(total=dl.n_data, dynamic_ncols=True, file=sys.stdout) as pbar:
            for i, (img, _, _, img_mask, img_angle) in enumerate(dl.load_cls()):
                
                img_mask = img_mask.to(torch.bool)
                colour_r, colour_g, colour_b, colour_a = img[0, :, img_mask].mean(dim=-1).tolist()
                location_x, location_y = center_of_mask(img_mask)
                
                fa_data['colour_r'].append(colour_r)
                fa_data['colour_g'].append(colour_g)
                fa_data['colour_b'].append(colour_b)
                fa_data['colour_a'].append(colour_a)
                fa_data['location_x'].append(location_x)
                fa_data['location_y'].append(location_y)
                fa_data['angle'].append(img_angle)
                angle, p1, p2 = angle_of_hull_of_mask(img_mask)
                fa_data['comp_angle'].append(angle)
                fa_data['sin2'].append(sin(2*angle))
                fa_data['cos2'].append(cos(2*angle))
                pbar.update(1)
                
        fa_data['colour_r'] = torch.Tensor(fa_data['colour_r'])
        fa_data['colour_g'] = torch.Tensor(fa_data['colour_g'])
        fa_data['colour_b'] = torch.Tensor(fa_data['colour_b'])
        fa_data['colour_a'] = torch.Tensor(fa_data['colour_a'])
        fa_data['location_x'] = torch.Tensor(fa_data['location_x'])
        fa_data['location_y'] = torch.Tensor(fa_data['location_y'])
        fa_data['angle'] = torch.Tensor(fa_data['angle'])
        fa_data['comp_angle'] = torch.Tensor(fa_data['comp_angle'])
        fa_data['sin2'] = torch.Tensor(fa_data['sin2'])
        fa_data['cos2'] = torch.Tensor(fa_data['cos2'])
        with open(f'./data/task_rep/{split}/data_location.pickle', 'wb') as handle:
            print(f'\n[ Writing {split} split... ]\n')
            pickle.dump(fa_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(f'\n[ All done ! :P ]\n')