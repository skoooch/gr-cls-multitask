from data_processing.data_loader_v2 import DataLoader
from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
import scipy.ndimage as ndi
from utils.parameters import Params

def center_of_mask(mask):

    mask = np.asarray(mask, dtype=bool)
    cy, cx = ndi.center_of_mass(mask)

    return cx.item(), cy.item()

mask = np.array([[False, False, False, False, False],
                 [False, True, True, True, False],
                 [False, True, True, True, False],
                 [False, False, False, False, False]])

cx, cy = center_of_mask(mask)
print(f"Center coordinates: ({cx}, {cy})")

if __name__ == "__main__":
    
    params = Params()
    
    dl = DataLoader(
        path=params.TRAIN_PATH,
        batch_size=16,
        return_mask=True
    )
    
    fa_data = {
        'colour_r': [],
        'colour_g': [],
        'colour_b': [],
        'colour_a': [],
        'location_x': [],
        'location_y': []
    }
    
    # Preprocess CLS
    with tqdm(total=400, dynamic_ncols=True, file=sys.stdout) as pbar:
        for i, (img, cls_map, label, img_mask) in enumerate(dl.load_cls()):
            
            img_mask = img_mask.to(np.bool)
            colour_r, colour_g, colour_b, colour_a = img[0, :, img_mask].mean(dim=-1)
            location_x, location_y = center_of_mask(img_mask)
            
            fa_data['colour_r'].append(colour_r)
            fa_data['colour_g'].append(colour_g)
            fa_data['colour_b'].append(colour_b)
            fa_data['colour_a'].append(colour_a)
            fa_data['location_x'].append(location_x)
            fa_data['location_y'].append(location_y)
            
            pbar.update(1)