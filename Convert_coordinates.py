# Sample coordinates from the 1280x1280 image
X1 = 200
Y1 = 300
X2 = 800
Y2 = 900

# Resize factors
resize_factor_x = 2480 / 1280
resize_factor_y = 4890 / 1280

# Convert coordinates to original image size
X1_orig = int(X1 * resize_factor_x)
Y1_orig = int(Y1 * resize_factor_y)
X2_orig = int(X2 * resize_factor_x)
Y2_orig = int(Y2 * resize_factor_y)

# Now you can plot the coordinates on the original image (2480x4890)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load your original image (2480x4890)
original_image = np.zeros((4890, 2480, 3))  # Replace this with your actual image

# Plot the image
plt.imshow(original_image)

# Create a Rectangle patch
rect = patches.Rectangle((X1_orig, Y1_orig), X2_orig - X1_orig, Y2_orig - Y1_orig, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the plot
plt.gca().add_patch(rect)

plt.show()
