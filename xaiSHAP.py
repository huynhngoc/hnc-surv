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

def preprocess_volumes(images, modality_index=0, volume_shape=(64, 64, 64)):
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
shap_values = np.array(explainer.shap_values(input_data, nsamples=1))

shap_values_slice = shap_values[0, :, slice_index, :, :, 0]
shap_values_slice_squeezed = np.squeeze(shap_values_slice)
print(f" shap_values_slice {shap_values_slice.shape} \n "
      f"shap_values_slice_squeezed {shap_values_slice_squeezed.shape}")

plt.imshow(shap_values_slice_squeezed, cmap='gray')
plt.title(f"SHAP values")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# original image slice
ax[0].imshow(slice_to_display, cmap='gray')
ax[0].set_title('Original Image Slice')

# SHAP values for the same slice
ax[1].imshow(shap_values_slice_squeezed, cmap='RdBu', vmin=-np.max(np.abs(shap_values_slice_squeezed)),
             vmax=np.max(np.abs(shap_values_slice_squeezed)))
ax[1].set_title('SHAP Values')
plt.show()

#  Pseudo RGB for visualization
slice_to_display_rgb = np.stack([slice_to_display]*3, axis=-1)
shap_values_slice_rgb = np.stack([shap_values_slice_squeezed]*3, axis=-1)

# The SHAP plotting function expects the first dimension to be the batch size
# and the last dimension to be channels
shap_values_batch = np.expand_dims(shap_values_slice_rgb, axis=0)
slice_to_display_batch = np.expand_dims(slice_to_display_rgb, axis=0)

shap.image_plot(shap_values_batch, slice_to_display_batch)