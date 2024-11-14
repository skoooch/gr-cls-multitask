
from tracemalloc import start
import torch
import tqdm
background_color=(188,188,188)
import numpy as np
import os
import h5py
import torch
import math
import numpy
import matplotlib.pyplot as plt
import torch
from skimage.draw import polygon, polygon_perimeter
import numpy as np

def fill_point(array, p, fill_value = 0):
    p //= 1
    p = p.astype(int)
    array[p[0], p[1], :] = fill_value
    print(p)
    # print(p[0] + 1,p[1])
    # array[p[0] + 1,p[1], :] = fill_value
    # array[p[0] + 1,p[1] + 1, :] = fill_value
    # array[p[0],p[1] + 1, :] = fill_value
    
def fill_parallelogram(array, p1,p2,p3,p4, fill_value=0):
    shapePoints = np.array([p1, p2, p3, p4])
    points_r, points_c = shapePoints[:, 0], shapePoints[:, 1]
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    array[perimeter_r, perimeter_c]
    #array[points_r, points_c] = fill_value
    array[interior_r, interior_c] = fill_value  
    
def fill_parallelogram_shadow(array, p1,p2,p3,p4, fill_value=0):
    shapePoints = np.array([p1, p2, p3, p4])
    points_r, points_c = shapePoints[:, 0], shapePoints[:, 1]
    interior_r, interior_c = np.clip(polygon(points_r, points_c), 0, 223)
    perimeter_r, perimeter_c = np.clip(polygon_perimeter(points_r, points_c), 0, 223)
    array[interior_r, interior_c] = fill_value 
    # array[perimeter_r, perimeter_c] = fill_value
    
def fill_shadow(array, p1, p2, p3, p4, shadow_dist, start_fill, end_fill, steps=100):
    # Ensure the array is in the right shape and data type for modification
    assert array.shape == (224, 224, 3), "Expected array of shape (224, 224, 3)"
    p1_s, p2_s, p3_s, p4_s = p1, p2, p3, p4
    # Convert start and end colors to torch tensors for interpolation
    start_fill = torch.tensor(start_fill, dtype=torch.float32)
    end_fill = torch.tensor(end_fill, dtype=torch.float32)
    
    # Convert points to tensors for interpolation
    p1, p2, p3, p4 = map(lambda p: torch.tensor(p, dtype=torch.float32), [p1, p2, p3, p4])
    p1_s, p2_s, p3_s, p4_s = map(lambda p: torch.tensor(p, dtype=torch.float32), [p1_s, p2_s, p3_s, p4_s])
    for point in [p1_s, p2_s, p3_s, p4_s]:
        point[0] += shadow_dist[0]
        point[1] -= shadow_dist[1]
    # Loop through steps to create gradient fill
    for i in range(steps + 1):
        # Interpolate the fill color
        fill_color = ((1 - i / steps) * end_fill + (i / steps) * start_fill).to(torch.long)

        # Interpolate points
        p1_i = (1 - i / steps) * p1_s + (i / steps) * p1
        p2_i = (1 - i / steps) * p2_s + (i / steps) * p2
        p3_i = (1 - i / steps) * p3_s + (i / steps) * p3
        p4_i = (1 - i / steps) * p4_s + (i / steps) * p4
        # Create integer coordinates for the quadrilateral at this step
        pts = torch.stack([p1_i, p2_i, p3_i, p4_i]).round().to(torch.int32)
        fill_parallelogram_shadow(array, np.array(p1_i), np.array(p2_i), np.array(p3_i), np.array(p4_i), fill_value=fill_color)

def gs_cofficient(v1, v2):
    return numpy.dot(v2, v1) / numpy.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():  
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def parallelogram_area(A, B, C, D):
    # Convert points to numpy arrays
    A, B, C, D = np.array(A), np.array(B), np.array(C), np.array(D)
    
    # Get two adjacent vectors AB and AD
    AB = B - A
    AD = D - A
    
    # Calculate the area using the cross product
    area = np.linalg.norm(np.cross(AB, AD))
    
    return area
# Define image dimensions and padding constraints
image_size = 224
max_rect_size = 120
min_rect_size = 30
cube_top_color = 255
cube_right_color = 245
cube_front_color = 235
cube_left_color = 195
cube_back_color = 205 
camera_height = 100
camera_tilt = 20
shadow_orderings = [cube_front_color, cube_left_color, cube_right_color, cube_back_color]
# Directory to save the images
output_dir = 'rectangle_dataset'
os.makedirs(output_dir, exist_ok=True)
filename = f'rect_data_angled.hdf5'
filepath = os.path.join(output_dir, filename)
h5_file = h5py.File(filepath, 'w')
angle_num = 8
data = h5_file.create_dataset('data', shape=(max_rect_size - min_rect_size, max_rect_size - min_rect_size, angle_num, 224, 224, 4), dtype=np.float32, fillvalue=188)
# Loop over all possible rectangle sizes
for width in range(119, max_rect_size):
    for height in range(30, max_rect_size):
        # Create a blank RGB image with a black background
        # Calculate top-left and bottom-right coordinates of the white rectangle
        x1 = (image_size - width) // 2
        y1 = (image_size - height) // 2
        x2 = x1 + width
        y2 = y1 + height
        z1 = 0
        z2 = 50
        cuboid = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        octagon_center = image_size//2
        octagon_y = camera_tilt / math.sqrt(2)
        camera_pos_list = [(octagon_y*1.5, -octagon_y*0.5),
                      (octagon_y*1.5, octagon_y*0.5),
                      (octagon_y*0.5, octagon_y*1.5),
                      (-octagon_y*0.5, octagon_y*1.5),
                      (-octagon_y*1.5, octagon_y*0.5),
                      (-octagon_y*1.5, -octagon_y*0.5),
                      (-octagon_y*0.5, -octagon_y*1.5),
                      (octagon_y*0.5, -octagon_y*1.5)]
        for i, camera_pos in enumerate(camera_pos_list):
            
            n = (camera_pos[0],camera_pos[1], camera_height)
            p = (octagon_center + camera_pos[0], octagon_center + camera_pos[1], camera_height)
            projected_cuboid = []
            for depth in [20, 0]:
                for point in cuboid:
                    t = (n[0]*p[0] - n[0]*point[0] + n[1]*p[1] - n[1]*point[1] + n[2]*p[2] - n[2]*depth) / (n[0]**2 + n[1]**2 + n[2]**2)
                    projj = (point[0] + t*n[0], point[1] + t*n[1], depth + t*n[2])
                    projected_cuboid.append(projj)
            middle_proj_t = (n[0]*p[0] - n[0]*octagon_center + n[1]*p[1] - n[1]*octagon_center + n[2]*p[2]) / (n[0]**2 + n[1]**2 + n[2]**2)
            middle_proj = (octagon_center + t*n[0], octagon_center + t*n[1], t*n[2])

            orthnorm_basis = gram_schmidt([np.array([projected_cuboid[3][0] - projected_cuboid[0][0],projected_cuboid[3][1] - projected_cuboid[0][1], projected_cuboid[3][2]- projected_cuboid[0][2]]), 
                                 np.array([projected_cuboid[1][0] - projected_cuboid[0][0],projected_cuboid[1][1]- projected_cuboid[0][1],projected_cuboid[1][2]- projected_cuboid[0][2]])])
            proj_back = []
            centroid_x = 0
            centroid_y = 0
            for j,point in enumerate(projected_cuboid):
                abc = np.array(point) - np.array(projected_cuboid[0])
                x,y = orthnorm_basis
                d = (x[1]*abc[0] - x[0]*abc[1])/(x[1]*y[0] - y[1]*x[0])
                t = (abc[0] - d*y[0])/x[0]
                proj_back.append((t,d))
                
                centroid_x += t
                centroid_y += d
            distance_x = image_size//2 - centroid_x/8
            distance_y = image_size//2 - centroid_y/8
            
            proj_back = np.array(proj_back)
            proj_back[:, 0] += distance_x
            proj_back[:, 1] += distance_y
            proj_back[:, 1] = image_size - proj_back[:, 1]
            # this is super convoluted as I really did not want to do actual lighting sim
            for j, shadow_vec in enumerate([(10,10), (10, -10), (-10, -10), (10, -10)]):
                img = torch.full((image_size, image_size, 3), 188)
                depth_img = torch.full((image_size, image_size, 1), -0.2)
                fill_shadow(img, proj_back[1], proj_back[3], proj_back[2], proj_back[0], shadow_vec, 120, 188)
                #front
                if 3 <= i and i <= 6: 
                    fill_parallelogram(img, proj_back[0], proj_back[1], proj_back[5], proj_back[4], shadow_orderings[(0 + j) % 4])
                #left
                if 1 <= i and i <= 4:
                    fill_parallelogram(img, proj_back[3], proj_back[1], proj_back[5], proj_back[7], shadow_orderings[(1 + j) % 4])
                #right         
                if 0 >= i or i >= 5:   
                    fill_parallelogram(img, proj_back[0], proj_back[2], proj_back[6], proj_back[4], shadow_orderings[(2 + j) % 4])
                # back
                if 2 >= i or i >= 7:
                    fill_parallelogram(img, proj_back[2], proj_back[3], proj_back[7], proj_back[6], shadow_orderings[(3 + j) % 4])
                fill_parallelogram(img, proj_back[0], proj_back[1], proj_back[3], proj_back[2], cube_top_color)
                fill_parallelogram(depth_img, proj_back[0], proj_back[1], proj_back[3], proj_back[2], 0.1)
                plt.imshow(img)
                plt.savefig(f"{j}_{i}.png")    
                plt.close()
                plt.clf()
                
                img = torch.cat((img, depth_img), dim=-1)
                data[width-30][height-30][i] = img
        exit()#right#back
h5_file.close()