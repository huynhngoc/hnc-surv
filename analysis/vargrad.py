import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


with h5py.File('C:/Users/Windows User/Documents/UNI/M30-DV/XAI/ous_test.h5', 'r') as f:
    data = f[f'{patient_index}'][:]

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

modalities = ['CT', 'PET', 'GTVp']
images_modalities = [ct_image, pet_image, gtv_image]
heatmaps = [heatmap_ct, heatmap_pet, heatmap_gtv]

# zoom region (x_start, x_end, y_start, y_end)
# original and first heatmap
zoom_region = (10, 210, 10, 200)

# zoomed heatmap
gtv_positions = np.where(gtv_image == 1)
x_min, x_max = np.min(gtv_positions[1]), np.max(gtv_positions[1])
y_min, y_max = np.min(gtv_positions[0]), np.max(gtv_positions[0])
margin = 20
zoom_region_gtv = (
    max(x_min - margin, 0), min(x_max + margin, gtv_image.shape[1]),
    max(y_min - margin, 0), min(y_max + margin, gtv_image.shape[0])
)

heatmap_min = min([heatmap.min() for heatmap in heatmaps])
heatmap_max = max([heatmap.max() for heatmap in heatmaps])

for i, (modality, image_modality, heatmap) in enumerate(zip(modalities, images_modalities, heatmaps)):
    fig, axs = plt.subplots(3, 1, figsize=(8, 24), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.patch.set_facecolor('black')

    # modality and cmap adjustments
    colors = [(0, 0, 0), (0, 1, 0)]  # Black to green
    n_bins = 100
    cmap_heat = LinearSegmentedColormap.from_list('BlackGreen', colors, N=n_bins)

    if modality == 'CT':
        vmin, vmax, cmap = 1024 + 70 - 100, 1024 + 70 + 100, 'gray'
    elif modality == 'PET':
        vmin, vmax, cmap = 0, 10, 'inferno'
    else:
        vmin, vmax, cmap = None, None, 'gray'

    # Original images with contour.
    im = axs[0].imshow(image_modality, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].contour(gtv_image, colors='cyan', linewidths=2)
    axs[0].axis('off')
    axs[0].set_xlim(zoom_region[0], zoom_region[1])
    axs[0].set_ylim(zoom_region[2], zoom_region[3])
    axs[0].invert_yaxis()

    # Images with heatmaps overlaid.
    axs[1].imshow(image_modality, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].contour(gtv_image, colors='cyan', linewidths=2)
    axs[1].imshow(heatmap, cmap=cmap_heat, alpha=0.7, vmin=heatmap_min, vmax=heatmap_max)
    axs[1].axis('off')
    axs[1].set_xlim(zoom_region[0], zoom_region[1])
    axs[1].set_ylim(zoom_region[2], zoom_region[3])
    axs[1].invert_yaxis()

    axs[2].imshow(image_modality, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].contour(gtv_image, colors='cyan', linewidths=2)
    axs[2].imshow(heatmap, cmap=cmap_heat, alpha=0.7, vmin=heatmap_min, vmax=heatmap_max)
    axs[2].axis('off')
    axs[2].set_xlim(zoom_region_gtv[0], zoom_region_gtv[1])
    axs[2].set_ylim(zoom_region_gtv[2], zoom_region_gtv[3])
    axs[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/VarGrad/OS_CT_PET_tumor_vargrad_{modality}.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 1))
    fig.patch.set_facecolor('white')

    cmap = plt.cm.hot
    #norm = plt.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_heat), cax=ax, orientation='horizontal')
    cb.ax.tick_params(colors='black')

    plt.tight_layout()
    plt.savefig('C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/VarGrad/OS_CT_PET_tumor_vargrad_colorbar.png', dpi=300, facecolor='white')

    plt.close()