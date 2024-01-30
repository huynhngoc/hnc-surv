import h5py
import numpy as np
import pandas as pd


def create_time_intervals_with_counts(data, num_intervals=21, num_equal_intervals=10, custom_breaks=None):
    # Sort the data by OS
    sorted_data = data.sort_values(by='OS')

    # Function to calculate counts for given breaks
    def calculate_counts_for_breaks(breaks):
        return [sorted_data[(sorted_data['OS'] > breaks[i]) & (sorted_data['OS'] <= breaks[i + 1])].shape[0]
                for i in range(len(breaks) - 1)]

    # Quantile-based intervals
    quantile_breaks = np.round(sorted_data['OS'].quantile(np.linspace(0, 1, num_intervals + 1)).values)
    quantile_breaks[0] = 0  # Ensure the first break is 0
    quantile_counts = calculate_counts_for_breaks(quantile_breaks)

    # Equally spaced intervals
    max_os = sorted_data['OS'].max()
    equal_breaks = np.round(np.linspace(0, max_os, num_equal_intervals + 1))
    equal_counts = calculate_counts_for_breaks(equal_breaks)

    results = {
        'quantile_breaks': quantile_breaks,
        'quantile_counts': quantile_counts,
        'equal_breaks': equal_breaks,
        'equal_counts': equal_counts
    }

    # Custom intervals
    if custom_breaks is not None:
        custom_breaks = np.round(custom_breaks)  # Round custom breaks
        if custom_breaks[0] != 0:
            custom_breaks[0] = 0  # Ensure the first break is 0
        custom_counts = calculate_counts_for_breaks(custom_breaks)
        results['custom_breaks'] = custom_breaks
        results['custom_counts'] = custom_counts

    return results


df = pd.read_csv('C:/Users/Windows User/Documents/UNI/M30-DV/HNC dataset/response_ous.csv',
                 delimiter=';')
# our first used break points [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
custom_break_points = [0, 6, 8, 12, 16,  20,  24,  32,  39,  42,  45,  49,  55,  59,  64, 67,  69,  76,  81,  85,  90]
results = create_time_intervals_with_counts(df, custom_breaks=custom_break_points)

# Print Quantile-based Break Points and Counts
print("Quantile-based Break Points:")
print(results['quantile_breaks'])
print("Quantile-based Patient Counts:")
print(results['quantile_counts'])

# Print Equally Spaced Break Points and Counts
print("\nEqually Spaced Break Points:")
print(results['equal_breaks'])
print("Equally Spaced Patient Counts:")
print(results['equal_counts'])

# Print Custom Break Points and Counts (if provided)
if 'custom_breaks' in results:
    print("\nCustom Break Points:")
    print(results['custom_breaks'])
    print("Custom Patient Counts:")
    print(results['custom_counts'])


# def transform_surv_data(breaks, data, targets):
#     t = targets[:, 1]
#     f = targets[:, 0]
#     n_samples = t.shape[0]
#     n_intervals = len(breaks) - 1
#     timegap = breaks[1:] - breaks[:-1]
#     breaks_midpoint = breaks[:-1] + 0.5 * timegap
#     y_train = np.zeros((n_samples, n_intervals * 2))
#     for i in range(n_samples):
#         if f[i]:  # if failed (not censored)
#             y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[1:])  # give credit for surviving each time interval where failure time >= upper limit
#             if t[i] < breaks[-1]:  # if failure time is greater than end of last time interval, no time interval will have failure marked
#                 y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][
#                     0]] = 1  # mark failure at first bin where survival time < upper break-point
#         else:  # if censored
#             y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)  # if censored and lived more than half-way through interval, give credit for surviving the interval.
#     # put original data in the end
#     return data, np.concatenate([y_train, targets], axis=-1)
#
# h5_filename = 'P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/HNC dataset/outcome_ous.h5'
#
# with h5py.File(h5_filename, 'r') as f:
#     images = f['fold_0']['image'][:4]
#     targets = f['fold_0']['OS_surv'][:4]
#
#
# X, y = transform_surv_data(np.arange(10, 101, 10), images, targets)
#
# exp = DefaultExperimentPipeline(
#         log_base_path='../../surv_log',
#         temp_base_path='../../surv_log_tmp'
#     ).from_full_config(
#         'config/survival_img_CT_PET_test_local/DFS_CT_PET_tmp_model_f01234_loglikelihood.json')
#
# import customize_obj
#
#
# exp.model.model.summary()
#
# data_gen = exp.model.data_reader.train_generator
# batch_x, batch_y = next(data_gen.generate())
#
# exp.model.model.fit(batch_x, batch_y)
#
# with h5py.File(h5_filename, 'r') as f:
#     print(f['fold_0'].keys())
#
# with h5py.File('../../surv_log/prediction/prediction.001.h5', 'r') as f:
#     print(f.keys())
#     for k in f.keys():
#         print(f[k])
