import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def box_count(img, box_sizes):
    """Computes the number of boxes needed to cover the pattern."""
    counts = []
    for size in box_sizes:
        S = size
        img_resized = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        non_empty_boxes = np.count_nonzero(img_resized)  # Count non-zero pixels
        counts.append(max(non_empty_boxes, 1))  # Avoid log(0)
    return counts

def preprocess_image(image_path):
    """Preprocess image to prepare it for fractal analysis."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarize the image using a global thresholding method
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Optional: Further refinement using closing operation to fill gaps (can enhance patterns)
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    return img

def fractal_dimension(image_path, box_sizes):
    """Computes the Fractal Dimension (FD) using the box-counting method."""
    img = preprocess_image(image_path)
    counts = box_count(img, box_sizes)
    
    log_sizes = np.log(box_sizes)
    log_counts = np.log(counts)
    
    # Linear fit to log-log data
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fd = abs(coeffs[0])  # Ensure FD is always positive
    
    return fd

# Define Paths
base_folder = r"C:\Users\HP\Bio Medical Analysis\Brain Tumor\Testing"
output_graph_path = os.path.join(base_folder, "fractal_dimensions_updated.png")

# Define box sizes for box-counting method
box_sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256])

# Process each category folder
fd_values = {}
for category in ["glioma", "meningioma", "notumor", "pituitary"]:
    category_path = os.path.join(base_folder, category)
    if not os.path.exists(category_path):
        continue
    
    for file in os.listdir(category_path):
        if file.endswith(".jpg"):
            image_path = os.path.join(category_path, file)
            fd = fractal_dimension(image_path, box_sizes)
            fd_values[file] = (fd, category)
            print(f"Fractal Dimension of {file} ({category}): {fd:.4f}")

# Sort results for better visualization
sorted_fd_values = sorted(fd_values.items(), key=lambda x: x[1][0])
image_names = [x[0] for x in sorted_fd_values]
fd_scores = [x[1][0] for x in sorted_fd_values]
categories = [x[1][1] for x in sorted_fd_values]

# Assign colors based on categories
category_colors = {"glioma": "red", "meningioma": "orange", "notumor": "green", "pituitary": "blue"}
colors = [category_colors[cat] for cat in categories]

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(image_names, fd_scores, color=colors)
plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Image")
plt.ylabel("Fractal Dimension (FD)")
plt.title("Fractal Dimensions of Brain Tumor Images (Updated)")
plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in category_colors.items()], loc='upper right')
plt.ylim(1, 3)  # ✅ Ensure positive Y-axis values
plt.tight_layout()
plt.savefig(output_graph_path)
print(f"✅ Fractal dimension plot saved at: {output_graph_path}")
plt.show()
