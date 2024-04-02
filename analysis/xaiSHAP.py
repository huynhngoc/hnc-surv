from deoxys.model import load_model
from skimage.transform import resize
import h5py
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import customize_obj


model_path = '/mnt/project/ngoc/hn_surv/perf/OS_PET_loglikelihood_f01234/model/model.032.h5'
model_test = 'C:/Users/Windows User/Documents/UNI/M30-DV/surv_perf/model/model.002.h5'
data_path = '../datasets/outcome_ous.h5'

model = load_model(model_path).model

print("model loaded")

with h5py.File(data_path, 'r') as hf:
    images = np.array(hf['fold_0']['image'])  # shape (28, 173, 191, 265, 4), (num_samples, depth, height, width, channels)


sample_index = 0
slice_index = 90 # 86 109 for sample 0

# find slices with tumor
sample_of_interest = images[sample_index]
slices_with_ones = []
# Loop through each slice in the GTV channel
for slice_index in range(sample_of_interest.shape[0]):  # Loop through the depth dimension
    current_slice = sample_of_interest[slice_index, ..., 2]
    if np.any(current_slice == 1):
        slices_with_ones.append(slice_index)

print(f"Slices with tumor: {slices_with_ones}")

images_with_three_channels = images[:, :, :, :, :3]
input_data = images_with_three_channels[sample_index][np.newaxis, :]

# {
#     "class_name": "HounsfieldWindowingPreprocessor",
#     "config": {
#         "window_center": 70,
#         "window_width": 200,
#         "channel": 0
#     }
# },
# {
#     "class_name": "ImageNormalizerPreprocessor",
#     "config": {
#         "vmin": [
#             -100,
#             0
#         ],
#         "vmax": [
#             100,
#             25
#         ]
#     }


# give all fold 4 samples as background set
explainer = shap.GradientExplainer(model, images_with_three_channels[:10], batch_size=5)

shap_values = np.array(explainer.shap_values(input_data, nsamples=1))

print(f'max shap value {np.max(shap_values)}')


print(f'shap value shape {shap_values.shape}')
# shap_values shape (10, 1, 173, 191, 265, 3), (timeIntervals, batch size, depth, height, width, channel)

for channel in range(3):  # Loop through each channel
    shap_values_channel = shap_values[0, :, slice_index, :, :, channel]
    shap_values_slice_squeezed = np.squeeze(shap_values_channel)
    original_image_channel = input_data[0, slice_index, :, :, channel]

    slice_to_display_rgb = np.stack([original_image_channel] * 3, axis=-1)
    shap_values_slice_rgb = np.stack([shap_values_slice_squeezed] * 3, axis=-1)

    shap_values_batch = np.expand_dims(shap_values_slice_rgb, axis=0)
    slice_to_display_batch = np.expand_dims(slice_to_display_rgb, axis=0)

    shap.image_plot(shap_values_batch, slice_to_display_batch)

    plt.savefig(f'shap_OS_CT_PET_tumor_channel{channel}_slice{slice_index}.png')
    plt.close()
