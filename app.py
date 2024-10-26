import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def reduce_colors_with_alpha(image_path, output_path, n_colors):
    image = Image.open(image_path).convert('RGBA')
    image_np = np.array(image)
    rgb_pixels = image_np[:, :, :3].reshape(-1, 3)
    alpha_channel = image_np[:, :, 3].reshape(-1)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(rgb_pixels)
    
    palette = np.round(kmeans.cluster_centers_).astype(int)
    labels = kmeans.labels_
    
    quantized_rgb_pixels = palette[labels]
    quantized_rgb_image_np = quantized_rgb_pixels.reshape(image_np.shape[:2] + (3,))
    
    quantized_image_np = np.dstack((quantized_rgb_image_np, alpha_channel.reshape(image_np.shape[:2])))
    
    quantized_image = Image.fromarray(quantized_image_np.astype('uint8'), 'RGBA')
    quantized_image.save(output_path)

    print(f"Image saved to {output_path} with {n_colors} colors (transparency preserved).")

if __name__ == "__main__":
    input_folder = "input image here"
    output_folder = "output image here"
    
    os.makedirs(output_folder, exist_ok=True)

    input_images = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    if not input_images:
        print("No PNG images found in the input folder.")
    else:
        for image_name in input_images:
            input_image_path = os.path.join(input_folder, image_name)
            output_image_path = os.path.join(output_folder, f"reduced_{image_name}")

            n_colors = int(input(f"Enter the number of colors to reduce '{image_name}' to: "))
            reduce_colors_with_alpha(input_image_path, output_image_path, n_colors)
