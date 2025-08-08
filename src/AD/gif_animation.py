import imageio.v2 as imageio
import os

# Directory containing your images
image_dir = "path/to/your/images"
images = []

# Make sure your images are sorted if necessary
file_names = sorted(os.listdir(image_dir))

for filename in file_names:
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_dir, filename)
        images.append(imageio.imread(image_path))

# Save as GIF
imageio.mimsave("output.gif", images, duration=0.2)  # duration in seconds per frame