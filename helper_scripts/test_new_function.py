import h5py
import numpy as np


def transform_surv_data(breaks, data, targets):
    t = targets[:, 1]
    f = targets[:, 0]
    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_samples, n_intervals * 2))
    for i in range(n_samples):
        if f[i]:  # if failed (not censored)
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[1:])  # give credit for surviving each time interval where failure time >= upper limit
            if t[i] < breaks[-1]:  # if failure time is greater than end of last time interval, no time interval will have failure marked
                y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][
                    0]] = 1  # mark failure at first bin where survival time < upper break-point
        else:  # if censored
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)  # if censored and lived more than half-way through interval, give credit for surviving the interval.
    # put original data in the end
    return data, np.concatenate([y_train, targets], axis=-1)

h5_filename = 'P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/HNC dataset/outcome_ous.h5'

with h5py.File(h5_filename, 'r') as f:
    images = f['fold_0']['image'][:4]
    targets = f['fold_0']['OS_surv'][:4]


X, y = transform_surv_data(np.arange(10, 101, 10), images, targets)

exp = DefaultExperimentPipeline(
        log_base_path='../../surv_log',
        temp_base_path='../../surv_log_tmp'
    ).from_full_config(
        'config/survival_img_CT_PET_test_local/DFS_CT_PET_tmp_model_f01234_loglikelihood.json')

import customize_obj


exp.model.model.summary()

data_gen = exp.model.data_reader.train_generator
batch_x, batch_y = next(data_gen.generate())

exp.model.model.fit(batch_x, batch_y)

with h5py.File(h5_filename, 'r') as f:
    print(f['fold_0'].keys())

with h5py.File('../../surv_log/prediction/prediction.001.h5', 'r') as f:
    print(f.keys())
    for k in f.keys():
        print(f[k])
