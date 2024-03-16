import numpy as np
import h5py
import matplotlib.pyplot as plt

def avg_filter_per_channel(data):
    return np.concatenate([
        [data],  # (0,0,0)
        [np.roll(data, 1, axis=i) for i in range(3)],  # (one 1)
        [np.roll(data, -1, axis=i) for i in range(3)],
        [np.roll(data, 1, axis=p) for p in [(0, 1), (0, 2), (1, 2)]],  # two 1s
        [np.roll(data, -1, p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data, (-1, 1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data, (1, -1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data, r, (0, 1, 2)) for r in [
            (1, 1, -1), (1, -1, 1), (-1, 1, 1),
            (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
            1, -1]
         ]
    ]).mean(axis=0)

def avg_filter(data):
    if len(data.shape) == 4 and data.shape[-1] > 1:
        return np.stack([avg_filter_per_channel(data[..., i]) for i in range(data.shape[-1])], axis=3)
    else:
        return avg_filter_per_channel(data)

patient_index = 189

with h5py.File('../../xai/ous_test.h5', 'r') as f:
    data = f['{patient_index}'][:]

with h5py.File('../../HNC dataset/outcome_ous.h5', 'r') as g:
    fold = g['fold_4']

    patient_indices = fold['patient_idx'][:]

    idx = list(patient_indices).index(patient_index)

    images = fold['image'][idx]

print(images.shape)
smooth_data = avg_filter(data)
smooth_data_v2 = avg_filter(smooth_data)

print(data.shape)

print(smooth_data_v2.shape)
# (173, 191, 265, 3) (depth, height, width, modality) 0-CT 1-PET 2-HEATMAP

slice_number = 132 # gtv from 126 to 147

ct_image = images[slice_number, :, :, 0]
pet_image = images[slice_number, :, :, 1]
gtv_image = images[slice_number, :, :, 2]

heatmap_ct = smooth_data_v2[slice_number, :, :, 0]
heatmap_pet = smooth_data_v2[slice_number, :, :, 1]
heatmap_gtv = smooth_data_v2[slice_number, :, :, 2]

modalities = ['CT', 'PET', 'GTVt']
images_modalities = [ct_image, pet_image, gtv_image]
heatmaps = [heatmap_ct, heatmap_pet, heatmap_gtv]


for i, (modality, image_modality, heatmap) in enumerate(zip(modalities, images_modalities, heatmaps)):
    fig, axs = plt.subplots(2, 1, figsize=(8, 16), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.patch.set_facecolor('black')

    # Original images
    axs[0].imshow(image_modality, cmap='gray')
    axs[0].contour(gtv_image, colors='lime', linewidths=2)
    axs[0].axis('off')

    # Images with heatmaps overlaid
    axs[1].imshow(image_modality, cmap='gray')
    axs[1].contour(gtv_image, colors='lime', linewidths=2)
    axs[1].imshow(heatmap, cmap='afmhot', alpha=0.7)
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/VarGrad/OS_CT_PET_tumor_vargrad_{modality}.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 1))
    fig.patch.set_facecolor('white')

    cmap = plt.cm.hot
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.ax.tick_params(colors='black')

    plt.tight_layout()
    plt.savefig('C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/VarGrad/OS_CT_PET_tumor_vargrad_colorbar.png', dpi=300, facecolor='white')

    plt.close()


