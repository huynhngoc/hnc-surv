import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import lifelines.datasets
from scipy.ndimage.measurements import center_of_mass
import matplotlib.patches as mpatches
import h5py


############################ KM CURVE #######################################

df = lifelines.datasets.load_g3()

df_male = df[df["sex"] == 'Male']
df_female = df[df["sex"] == 'Female']


time_male = df_male['time']
event_male = df_male['event']

time_female = df_female['time']
event_female = df_female['event']

kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()

# Fit the data
kmf1.fit(time_male, event_male, label='Group 1')
kmf2.fit(time_female, event_female, label='Group 2')


plt.figure(figsize=(8, 8), dpi=600)
ax = kmf1.plot_survival_function(ci_show=False, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})
kmf2.plot_survival_function(ax=ax, ci_show=False, show_censors=True, censor_styles={'marker': '+', 'ms': 10, 'mew': 1.5})

add_at_risk_counts(kmf1, kmf2, ax=ax)

plt.title('Example KM curve')
plt.tight_layout()
ax.legend(loc='lower left', fontsize=12)
ax.set_xlabel('Time Intervals', fontsize=12)
ax.set_ylabel('Survival probability', fontsize=12)
plt.show()
plt.close()

######################### ROC "#######################################################################################
sns.set_style("darkgrid")

np.random.seed(123)
y_true = np.random.randint(0, 2, 100)
y_scores = np.random.uniform(0.5, 0.1, 100) + (0.1 * y_true)

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(8, 6), dpi=600)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Performance')
plt.plot([0, 0, 1], [0, 1, 1], color='green', linestyle='--', lw=2, label='Perfect Performance')

plt.fill_between(fpr, tpr, color='skyblue', alpha=0.2)
plt.plot([], [], color='skyblue', alpha=0.5, label='AUC - shaded area')

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc="lower right", frameon=True, fontsize=14)
plt.show()

################################### image modalities ##########################
sns.set_style("whitegrid")
patient_index = 50
slice_number = 102

with h5py.File('C:/Users/Windows User/Documents/UNI/M30-DV/HNC dataset/outcome_ous.h5', 'r') as g:
    fold = g['fold_2']

    patient_indices = fold['patient_idx'][:]
    print(patient_indices)

    idx = list(patient_indices).index(patient_index)

    images = fold['image'][idx]


ct_image = images[slice_number, :, :, 0]
pet_image = images[slice_number, :, :, 1]
gtvp_image = images[slice_number, :, :, 2]
gtvn_image = images[slice_number, :, :, 3]


def find_centered_nonzero_bounds(image, size=(200, 200)):
    """
    Finds a zoom region centered around the center of mass of non-zero pixels.

    Parameters:
    - image: The image array.
    - size: The desired size of the zoom region (height, width).

    Returns:
    - A tuple defining the zoom region (xmin, xmax, ymin, ymax).
    """
    # Calculate the center of mass for non-zero pixels
    center_y, center_x = center_of_mass(image > 0)  # Adjust condition as needed

    # Define half sizes for easier calculation
    half_height = size[0] // 2
    half_width = size[1] // 2

    # Calculate bounds, ensuring they are within the image dimensions
    ymin = max(0, int(center_y) - half_height)
    ymax = min(image.shape[0], int(center_y) + half_height)
    xmin = max(0, int(center_x) - half_width)
    xmax = min(image.shape[1], int(center_x) + half_width)

    return (xmin, xmax, ymin, ymax)

# Apply the function to the CT image to get a centered zoom_region
zoom_region = find_centered_nonzero_bounds(ct_image, size=(170, 170))

slices_with_1 = []
for slice_number in range(images[:, :, :, 3].shape[0]):
    if np.any(images[slice_number, :, :, 3] == 1) and np.any(images[slice_number, :, :, 2] == 1):
        slices_with_1.append(slice_number)
print("Slices with both tumor and node:", slices_with_1)

images_modalities = [ct_image, pet_image, gtvp_image, gtvn_image]
modalities = ['CT', 'PET', 'GTVp', 'GTVn']

for i, (modality, image_modality) in enumerate(zip(modalities, images_modalities)):
    fig, axs = plt.subplots(1, 1, figsize=(8, 8), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.patch.set_facecolor('black')

    if modality == 'CT':
        vmin, vmax, cmap = 1024 + 70 - 100, 1024 + 70 + 100, 'gray'
    elif modality == 'PET':
        vmin, vmax, cmap = 0, 10, 'inferno'
    else:
        vmin, vmax, cmap = None, None, 'gray'

    # Original images with contour.

    im = axs.imshow(image_modality, cmap=cmap, vmin=vmin, vmax=vmax)
    if modality == 'CT' or modality == 'PET':
        axs.contour(gtvp_image, colors='cyan', linewidths=2)
        axs.contour(gtvn_image, colors='pink', linewidths=2)
    axs.axis('off')
    axs.set_xlim(zoom_region[0], zoom_region[1])
    axs.set_ylim(zoom_region[3], zoom_region[2])
    axs.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/examples/{modality}.png', dpi=600)
    plt.close()


proxy_gtvp = mpatches.Patch(color='cyan', label='GTVp')
proxy_gtvn = mpatches.Patch(color='pink', label='GTVn')

fig_legend = plt.figure(figsize=(1, 1))
ax_legend = fig_legend.add_subplot(111)
legend = ax_legend.legend(handles=[proxy_gtvp, proxy_gtvn], loc='center')
ax_legend.axis('off')
fig_legend.savefig("C:/Users/Windows User/Documents/UNI/M30-DV/avhandling/m30-dv/images/examples/legend.png", bbox_inches='tight', dpi=600)
