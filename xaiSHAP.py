import customize_obj
from deoxys.model import load_model
from skimage.transform import resize
import h5py
import numpy as np
import shap
import matplotlib.pyplot as plt


model_path = 'C:/Users/Windows User/Documents/UNI/M30-DV/ulrik/stuk/model.055.h5'
model_test = 'C:/Users/Windows User/Documents/UNI/M30-DV/surv_perf/model/model.002.h5'
data_path = 'C:/Users/Windows User/Documents/UNI/M30-DV/HNC dataset/outcome_ous.h5'

model = load_model(model_test).model

print("model loaded")

with h5py.File(data_path, 'r') as hf:
    images = np.array(hf['fold_4']['image'])  # shape (28, 173, 191, 265, 4), (num_samples, depth, height, width, channels)

def preprocess_volumes(images, modality_index=1, volume_shape=(64, 64, 64)):
    """Extract a 3D volume from 4D image volumes.
    Args:
        images: 5D numpy array of shape (num_samples, depth, height, width, channels).
        modality_index: Index of the modality to extract (0: CT, 1: PET, 2: GTVt, 3: GTVn).
        volume_shape: Desired output shape of the volume (depth, height, width).
    Returns:
        5D numpy array of processed volumes of shape (num_samples, depth, height, width, 1).
    """
    processed_volumes = []
    for img in images:
        # Extract the modality across all slices
        modality_volume = img[:, :, :, modality_index]
        # Select middle 64 slices
        mid_start = (modality_volume.shape[0] - volume_shape[0]) // 2
        mid_volume = modality_volume[mid_start:mid_start + volume_shape[0]]
        resized_volume = np.zeros(volume_shape)
        for i, slice in enumerate(mid_volume):
            resized_volume[i] = resize(slice, (volume_shape[1], volume_shape[2]),
                                       anti_aliasing=True, mode='reflect')
        # Expand dimensions to add the channel dimension
        processed_volumes.append(np.expand_dims(resized_volume, axis=-1))
    return np.array(processed_volumes)

volumes = preprocess_volumes(images)

sample_index = 0
slice_index = 32
slice_to_display = volumes[sample_index, slice_index, :, :, 0]
plt.imshow(slice_to_display, cmap='gray')
plt.title(f"Sample {sample_index}, Slice {slice_index}")
plt.show()

print(f"volumes[0] {volumes[0].shape} \n slice_to_display {slice_to_display.shape}")

input_data = volumes[0][np.newaxis, :]
explainer = shap.GradientExplainer(model, volumes[0], batch_size=1)

# shap_values shape (10, 1, 64, 64, 64, 1), (timeIntervals, batch size, depth, height, width, channel)
shap_values = np.array(explainer.shap_values(input_data, nsamples=100))

shap_values_slice = shap_values[0, :, slice_index, :, :, 0]
shap_values_slice_squeezed = np.squeeze(shap_values_slice)
print(f" shap_values_slice {shap_values_slice.shape} \n "
      f"shap_values_slice_squeezed {shap_values_slice_squeezed.shape}")

plt.imshow(shap_values_slice_squeezed, cmap='gray')
plt.title(f"Gradient SHAP values")
plt.show()

#  Pseudo RGB for visualization
slice_to_display_rgb = np.stack([slice_to_display]*3, axis=-1)
shap_values_slice_rgb = np.stack([shap_values_slice_squeezed]*3, axis=-1)

# The SHAP plotting function expects the first dimension to be the batch size
# and the last dimension to be channels
shap_values_batch = np.expand_dims(shap_values_slice_rgb, axis=0)
slice_to_display_batch = np.expand_dims(slice_to_display_rgb, axis=0)

shap.image_plot(shap_values_batch, slice_to_display_batch)

###### Using KernelExplainer instead of GradientExplainer #######

# The KernelExplainer requires a reference (background) dataset. For simplicity, we can use the training dataset's mean.
# However, for more accurate explanations, you should use a representative subset of your data.
background = volumes.mean(axis=0)

background_2d = background.reshape(1, -1)  # Flatten the background to have shape (1, num_features)
input_data_2d = input_data.reshape(1, -1)  # Flatten the input data to have shape (1, num_features)

num_background_samples = 1  # For example, taking 50 samples for the background
sample_shape = volumes.shape[1:]  # This gets the shape of a single sample (depth, height, width, channels)
num_features = np.prod(sample_shape)  # This is the total number of features per sample

# Now reshape the background
background = volumes[:num_background_samples].reshape(num_background_samples, num_features)

# Define the wrapper function for predict
def model_predict_5d(input_data_2d):
    input_data_5d = input_data_2d.reshape(-1, 64, 64, 64, 1)  # Reshape back to (None, 64, 64, 64, 1)
    return model.predict(input_data_5d)

# Initialize the explainer with the new background
explainer = shap.KernelExplainer(model_predict_5d, background)

# Compute SHAP values
shap_values_kernel = explainer.shap_values(input_data_2d, nsamples=1)


# Reshape SHAP values back to original shape for plotting
shap_values_3d = shap_values_kernel[0].reshape(-1, 64, 64, 64, 1)

# Assuming you're working with the slice_index that you've previously defined
shap_values_slice_3d = shap_values_3d[:, slice_index, :, :, :]
shap_values_slice_3d_squeezed = np.squeeze(shap_values_slice_3d)
shap_values_slice_3d_rgb = np.stack([shap_values_slice_3d_squeezed]*3, axis=-1)

shap_values_batch_3d = np.expand_dims(shap_values_slice_3d_rgb, axis=0)


shap.image_plot(shap_values_batch_3d, slice_to_display_batch)

plt.imshow(shap_values_slice_3d_squeezed, cmap='gray')
plt.title(f"Kernel SHAP values")
plt.show()
