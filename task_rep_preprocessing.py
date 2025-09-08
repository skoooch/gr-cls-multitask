from data_processing.data_loader_v2 import DataLoader
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
from utils.parameters import Params
import torch
import pickle

def center_of_mask(mask):

    mask = np.asarray(mask, dtype=bool)
    cy, cx = ndi.center_of_mass(mask)

    return cx.item(), cy.item()

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
            return_mask=True
        )
        
        fa_data = {
            'colour_r': [],
            'colour_g': [],
            'colour_b': [],
            'colour_a': [],
            'location_x': [],
            'location_y': [],
            'img': []
        }
        
        # Preprocess CLS
        with tqdm(total=dl.n_data, dynamic_ncols=True, file=sys.stdout) as pbar:
            for i, (img, _, _, img_mask) in enumerate(dl.load_cls()):
                
                img_mask = img_mask.to(torch.bool)
                colour_r, colour_g, colour_b, colour_a = img[0, :, img_mask].mean(dim=-1).tolist()
                location_x, location_y = center_of_mask(img_mask)
                
                fa_data['colour_r'].append(colour_r)
                fa_data['colour_g'].append(colour_g)
                fa_data['colour_b'].append(colour_b)
                fa_data['colour_a'].append(colour_a)
                fa_data['location_x'].append(location_x)
                fa_data['location_y'].append(location_y)
                
                pbar.update(1)
                
        fa_data['colour_r'] = torch.Tensor(fa_data['colour_r'])
        fa_data['colour_g'] = torch.Tensor(fa_data['colour_g'])
        fa_data['colour_b'] = torch.Tensor(fa_data['colour_b'])
        fa_data['colour_a'] = torch.Tensor(fa_data['colour_a'])
        fa_data['location_x'] = torch.Tensor(fa_data['location_x'])
        fa_data['location_y'] = torch.Tensor(fa_data['location_y'])
        
        with open(f'./data/task_rep/{split}/data.pickle', 'wb') as handle:
            print(f'\n[ Writing {split} split... ]\n')
            pickle.dump(fa_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(f'\n[ All done ! :P ]\n')