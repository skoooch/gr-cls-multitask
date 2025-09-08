import os
from PIL import Image

# Set your directory and participant range
img_dir = "vis/rsm_correlation_1/single"
n_participants = 20  # adjust as needed
skip = [2, 10, 13, 17]  # participants to skip

# Collect images
class_imgs = []
grasp_imgs = []
indices = []
for i in range(1, n_participants + 1):
    if i in skip:
        continue
    class_path = os.path.join(img_dir, f"class_25_{i}.png")
    grasp_path = os.path.join(img_dir, f"grasp_25_{i}.png")
    if os.path.exists(class_path) and os.path.exists(grasp_path):
        class_imgs.append(Image.open(class_path))
        grasp_imgs.append(Image.open(grasp_path))
        indices.append(i)

# Assume all images are the same size
w, h = class_imgs[0].size
combined_img = Image.new('RGB', (w * 2 + 300, h * len(class_imgs)), color=(255,255,255))

# Paste images into grid
for idx, (c_img, g_img) in enumerate(zip(class_imgs, grasp_imgs)):
    combined_img.paste(c_img, (300, idx * h))
    combined_img.paste(g_img, (w + 300, idx * h))

# Optionally, add participant numbers as text (requires PIL.ImageDraw)
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(combined_img)
font = None
try:
    font = ImageFont.truetype("arial.ttf", 50)
except:
    font = ImageFont.load_default(40)
for idx, i in enumerate(indices):
    draw.text((10, idx * h + h/2), f"Participant {i}", fill=(0,0,0), font=font)

# Save the combined image
combined_img.save("all_participants_rsa.png")
print("Combined image saved as all_participants_rsa.png")