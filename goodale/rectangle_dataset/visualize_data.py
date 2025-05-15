import os
import subprocess
import matplotlib.pyplot as plt
import glob
import h5py
import numpy as np
def generate_video(img):
    for i in range(90):
        p = np.array(img[89 - i, i//2, 0, 0, :, :, :], dtype = np.uint8)
        plt.imsave("file%02d.png" % i, p)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name2.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
output_dir = '/scratch/expires-2024-Dec-14/'
os.makedirs(output_dir, exist_ok=True)
filename = f'rect_data_angled.hdf5'
filepath = os.path.join(output_dir, filename)
h5_file = h5py.File(filepath, 'r')

data = h5_file.get("data")

generate_video(data)