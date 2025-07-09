import numpy as np
from PIL import Image

def inspect_unique_labels(label_map_path):
    label_map = Image.open(label_map_path)
    label_array = np.array(label_map)
    unique_values = np.unique(label_array)
    print(f"Unique label IDs in image: {unique_values}")
    return unique_values

# Example usage
inspect_unique_labels("label_map.png")