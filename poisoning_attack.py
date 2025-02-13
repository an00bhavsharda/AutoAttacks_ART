import numpy as np
import matplotlib.pyplot as plt
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.utils import to_categorical

# Step 1: Define a Simple Backdoor Function
def backdoor_function(image):
    """
    Injects a small white square (backdoor) in the bottom-right corner of the image.
    """
    poisoned_image = image.copy()
    h, w, c = poisoned_image.shape  # Get height, width, and channels

    # Define the backdoor patch (small white square)
    patch_size = h // 5  # 1/5th of the image size
    poisoned_image[-patch_size:, -patch_size:, :] = 1.0  # Set patch to white (max pixel value)

    return poisoned_image

# Step 2: Initialize ART Poisoning Attack
backdoor = PoisoningAttackBackdoor(backdoor_function)

# Step 3: Generate a Clean Image (Simulated Dataset)
image_size = (32, 32, 3)  # CIFAR-10 size (32x32 RGB)
clean_image = np.random.rand(*image_size)  # Generate a random image
label = np.array([1])  # Example target label

# Step 4: Apply the Backdoor to Create a Poisoned Image
poisoned_image, poisoned_label = backdoor.poison(clean_image, y=label)

# Step 5: Visualize the Original and Poisoned Images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Original Image
ax[0].imshow(clean_image)
ax[0].set_title("Original Image")
ax[0].axis("off")

# Poisoned Image
ax[1].imshow(poisoned_image)
ax[1].set_title("Poisoned Image with Backdoor")
ax[1].axis("off")

plt.show()

print(f"✅ Poisoning attack completed! The image has been modified with a backdoor.\nTarget label: {poisoned_label}")
