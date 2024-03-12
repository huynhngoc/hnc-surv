import numpy as np
import h5py
import argparse
import pandas as pd


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
    if len(data.shape)==4 and data.shape[-1] > 1:
        return np.stack([avg_filter_per_channel(data[..., i]) for i in range(data.shape[-1])], axis=3)
    else:
        return avg_filter_per_channel(data)


def edge_detection(data):
    data_neg = 0 - data
    return np.concatenate([
        [data] * 26,  # (0,0,0)
        [np.roll(data_neg, 1, axis=i) for i in range(3)],  # (one 1)
        [np.roll(data_neg, -1, axis=i) for i in range(3)],
        [np.roll(data_neg, 1, axis=p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, -1, p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, (-1, 1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, (1, -1), p) for p in [(0, 1), (0, 2), (1, 2)]],
        [np.roll(data_neg, r, (0, 1, 2)) for r in [
                (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                (-1, -1, 1), (-1, 1, -1), (1, -1, -1),
            1, -1]
         ]
    ]).mean(axis=0)


def get_overall_info(data):
    print('Getting basic statistical information')
    return {
        'ct_total': (data[..., 0] > 0).sum(),
        'ct_sum': data[..., 0].sum(),
        'ct_max': data[..., 0].max(),
        'ct_mean': data[..., 0].mean(),
        'ct_std': data[..., 0].std(),
        'pt_total': (data[..., 1] > 0).sum(),
        'pt_sum': data[..., 1].sum(),
        'pt_max': data[..., 1].max(),
        'pt_mean': data[..., 1].mean(),
        'pt_std': data[..., 1].std(),
        'T_total': (data[..., 2] > 0).sum(),
        'T_sum': data[..., 2].sum(),
        'T_max': data[..., 2].max(),
        'T_mean': data[..., 2].mean(),
        'T_std': data[..., 2].std()
    }


def get_area_info(data, area, name):
    print('Getting information based information of', name)
    selected_data = data[area > 0]
    try:
        return {
            f'ct_{name}_total': (selected_data[..., 0] > 0).sum(),
            f'ct_{name}_sum': selected_data[..., 0].sum(),
            f'ct_{name}_max': selected_data[..., 0].max(),
            f'ct_{name}_mean': selected_data[..., 0].mean(),
            f'ct_{name}_std': selected_data[..., 0].std(),
            f'ct_{name}_q1': np.quantile(selected_data[..., 0], 0.25),
            f'ct_{name}_q2': np.quantile(selected_data[..., 0], 0.5),
            f'ct_{name}_q3': np.quantile(selected_data[..., 0], 0.75),
            f'pt_{name}_total': (selected_data[..., 1] > 0).sum(),
            f'pt_{name}_sum': selected_data[..., 1].sum(),
            f'pt_{name}_max': selected_data[..., 1].max(),
            f'pt_{name}_mean': selected_data[..., 1].mean(),
            f'pt_{name}_std': selected_data[..., 1].std(),
            f'pt_{name}_q1': np.quantile(selected_data[..., 1], 0.25),
            f'pt_{name}_q2': np.quantile(selected_data[..., 1], 0.5),
            f'pt_{name}_q3': np.quantile(selected_data[..., 1], 0.75),
            f'T_{name}_total': (selected_data[..., 2] > 0).sum(),
            f'T_{name}_sum': selected_data[..., 2].sum(),
            f'T_{name}_max': selected_data[..., 2].max(),
            f'T_{name}_mean': selected_data[..., 2].mean(),
            f'T_{name}_std': selected_data[..., 2].std(),
            f'T_{name}_q1': np.quantile(selected_data[..., 2], 0.25),
            f'T_{name}_q2': np.quantile(selected_data[..., 2], 0.5),
            f'T_{name}_q3': np.quantile(selected_data[..., 2], 0.75),
        }
    except Exception as e:
        print(e)
        print(f'Area {name} is empty')
        return {
            f'ct_{name}_total': 0,
            f'ct_{name}_sum': 0,
            f'ct_{name}_max': 0,
            f'ct_{name}_mean': 0,
            f'ct_{name}_std': 0,
            f'ct_{name}_q1': 0,
            f'ct_{name}_q2': 0,
            f'ct_{name}_q3': 0,
            f'pt_{name}_total': 0,
            f'pt_{name}_sum': 0,
            f'pt_{name}_max': 0,
            f'pt_{name}_mean': 0,
            f'pt_{name}_std': 0,
            f'pt_{name}_q1': 0,
            f'pt_{name}_q2': 0,
            f'pt_{name}_q3': 0,
            f'T_{name}_total': 0,
            f'T_{name}_sum': 0,
            f'T_{name}_max': 0,
            f'T_{name}_mean': 0,
            f'T_{name}_std': 0,
            f'T_{name}_q1': 0,
            f'T_{name}_q2': 0,
            f'T_{name}_q3': 0,
        }


def get_histogram_info(data, areas, names):
    print('Getting distribution data of', names)
    objs = {}
    for (area, name) in zip(areas, names):
        print(name)
        selected_data = data[area > 0]
        if (area > 0).sum():
            objs.update({
                f'{name}_area': (area > 0).sum(),
                f'{name}_total': (selected_data > 0).sum(),
                f'{name}_sum': selected_data.sum(),
                f'{name}_max': selected_data.max(),
                f'{name}_mean': selected_data.mean(),
                f'{name}_std': selected_data.std(),
                f'{name}_median': np.median(selected_data),
            })
        else:
            objs.update({
                f'{name}_area': (area > 0).sum(),
                f'{name}_total': 0,
                f'{name}_sum': 0,
                f'{name}_max': 0,
                f'{name}_mean': 0,
                f'{name}_std': 0,
                f'{name}_median': 0,
            })
    return objs


def get_info(data_normalized, ct_img, pt_img, tumor, node):
    # start the count
    overall_info = get_overall_info(data_normalized)

    # vargrad in tumour & node
    tumor_info = get_area_info(data_normalized, tumor, 'tumor')
    node_info = get_area_info(data_normalized, node, 'node')
    normal_voxel_info = get_area_info(data_normalized,
                                      1 - tumor - node,
                                      'outside')
    # correlation between intensity and vargrad
    suv_corr = np.corrcoef(
        pt_img.flatten(), data_normalized[..., 1].flatten())[0, 1]
    hu_corr = np.corrcoef(
        ct_img.flatten(), data_normalized[..., 0].flatten())[0, 1]

    # histogram data
    suv_zero = (pt_img < 0.04).astype(int)
    suv_0_2 = (pt_img <= 0.08).astype(int) - (pt_img < 0.04).astype(int)
    suv_2_4 = (pt_img <= 0.16).astype(int) - (pt_img <= 0.08).astype(int)
    suv_4_6 = (pt_img <= 0.24).astype(int) - (pt_img <= 0.16).astype(int)
    suv_6_8 = (pt_img <= 0.32).astype(int) - (pt_img <= 0.24).astype(int)
    suv_8_10 = (pt_img <= 0.4).astype(int) - (pt_img <= 0.32).astype(int)
    suv_10_over = (pt_img > 0.4).astype(int)

    areas = [suv_zero, suv_0_2, suv_2_4,
             suv_4_6, suv_6_8, suv_8_10, suv_10_over]
    area_names = ['suv_zeros', 'suv_0_2', 'suv_2_4', 'suv_4_6',
                  'suv_6_8', 'suv_8_10', 'suv_10_over']
    suv_info = get_histogram_info(data_normalized[..., 1], areas, area_names)

    all_info = {
        **overall_info,
        **tumor_info,
        **node_info,
        **normal_voxel_info,
        'hu_corr': hu_corr,
        'suv_corr': suv_corr,
        **suv_info
    }

    return all_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file")
    parser.add_argument("log_folder")
    parser.add_argument("--idx", default=0, type=int)

    args, unknown = parser.parse_known_args()

    base_folder = args.log_folder
    h5_file = args.h5_file

    print(f'Checking folder {base_folder}')

    if 'ous' in h5_file:
        center = 'OUS'
        print('Getting OUS dataset file...')
        src_data_file = '../datasets/outcome_ous.h5'
        curr_fold = f'fold_{base_folder[-1]}'
        with h5py.File(src_data_file, 'r') as f:
            pids = f[curr_fold]['patient_idx'][:]
    elif 'maastro' in h5_file:
        center = 'MAASTRO'
        src_data_file = '../datasets/outcome_maastro.h5'
        with h5py.File(src_data_file, 'r') as f:
            print('Getting MAASTRO dataset file')
            pids = []
            for key in f.keys():
                pids.extend([
                    pid for pid in f[key]['patient_idx'][:]
                ])
                if len(pids) > args.idx:
                    curr_fold = key
                    break
    print('List of PIDS:', pids)
    if args.idx < len(pids):
        pid = pids[args.idx]
        print('Patients:', pid)
        print('Getting image data...')
        with h5py.File(src_data_file, 'r') as f:
            img_idx = [pid
                       for pid in f[curr_fold]['patient_idx'][:]].index(pid)
            img = f[curr_fold]['image'][img_idx]
            dfs = f[curr_fold]['DFS'][img_idx]
            os = f[curr_fold]['OS'][img_idx]
            dfs_time = f[curr_fold]['DFS_surv'][img_idx][1]
            os_time = f[curr_fold]['OS_surv'][img_idx][1]

        print('Windowing CT images')
        # windowing
        ct_img = img[..., 0] - (1024 + 70)
        ct_img = ((ct_img.clip(-100, 100) + 100) / 200.).clip(0, 1)

        print('Normalize PET images')
        # clipped SUV
        pt_img = (img[..., 1] / 25.).clip(0, 1)

        tumor = img[..., 2]
        node = img[..., 3]

        # histogram data
        suv_zero = (pt_img < 0.04).astype(int)
        suv_0_2 = (pt_img <= 0.08).astype(int) - (pt_img < 0.04).astype(int)
        suv_2_4 = (pt_img <= 0.16).astype(int) - (pt_img <= 0.08).astype(int)
        suv_4_6 = (pt_img <= 0.24).astype(int) - (pt_img <= 0.16).astype(int)
        suv_6_8 = (pt_img <= 0.32).astype(int) - (pt_img <= 0.24).astype(int)
        suv_8_10 = (pt_img <= 0.4).astype(int) - (pt_img <= 0.32).astype(int)
        suv_10_over = (pt_img > 0.4).astype(int)

        areas = [suv_zero, suv_0_2, suv_2_4, suv_4_6,
                 suv_6_8, suv_8_10, suv_10_over]
        area_names = ['all_suv_zeros', 'all_suv_0_2', 'all_suv_2_4', 'all_suv_4_6',
                      'all_suv_6_8', 'all_suv_8_10', 'all_suv_10_over']

        print('Getting interpret resutls...')
        with h5py.File(args.log_folder + '/' + h5_file, 'r') as f:
            data = f[str(pid)][:]

        # # normalize original data
        # d_min = data.min()
        # d_max = data.max()

        # d_norm = ((data - d_min) / (d_max - d_min)).clip(0, 1)

        # basic_info = {
        #     'pid': pid,
        #     'center': center,
        #     'dfs': dfs,
        #     'os': os,
        #     'val_fold': int(base_folder[-2]),
        #     'test_fold': int(base_folder[-1]),
        #     'vargrad_sum': d_norm.sum(),
        #     'vargrad_ct_sum': d_norm[..., 0].sum(),
        #     'vargrad_pt_sum': d_norm[..., 1].sum(),
        #     'vargrad_T_sum': d_norm[..., 2].sum(),
        #     'hu_corr_all': np.corrcoef(d_norm[..., 0].flatten(),
        #                                ct_img.flatten())[0, 1],
        #     'suv_corr_all': np.corrcoef(d_norm[..., 1].flatten(),
        #                                 pt_img.flatten())[0, 1],
        #     'hu_corr_raw': np.corrcoef(d_norm[..., 0].flatten(),
        #                                img[..., 0].flatten())[0, 1],
        #     'suv_corr_raw': np.corrcoef(d_norm[..., 1].flatten(),
        #                                 img[..., 1].flatten())[0, 1],
        #     'tumor_size': (tumor > 0).sum(),
        #     **get_area_info(d_norm, tumor, 'tumor_all'),
        #     'node_size': (node > 0).sum(),
        #     **get_area_info(d_norm, node, 'node_all'),
        #     **get_area_info(d_norm, 1 - tumor - node, 'outside_all'),
        #     **get_histogram_info(d_norm[..., 1], areas, area_names)
        # }

        # raw_info = []
        # for quantile in [.95, .96, .97, .98, .99]:
        #     thres = np.quantile(data, quantile)
        #     max_vargrad = data.max()

        #     print('Normalizing interpret results...')
        #     data_normalized = (
        #         (data - thres) / (max_vargrad - thres)).clip(0, 1)
        #     raw_info.append({
        #         **basic_info,
        #         'quantile': quantile,
        #         'vargrad_max': max_vargrad,
        #         'vargrad_threshold': thres,
        #         'vargrad_sum_selected': data_normalized.sum(),
        #         **get_info(data_normalized, ct_img, pt_img, tumor, node),
        #     })

        # print('Saving raw resutls...')
        # pd.DataFrame(raw_info).to_csv(
        #     base_folder + f'/{center}/raw/{pid}.csv', index=False)

        print('Smoothening interpret results...')
        smoothen_data = avg_filter(data)

        # # normalize original data
        # s_d_min = smoothen_data.min()
        # s_d_max = smoothen_data.max()

        # s_d_norm = ((smoothen_data - s_d_min) / (s_d_max - s_d_min)).clip(0, 1)

        # s_basic_info = {
        #     'pid': pid,
        #     'center': center,
        #     'dfs': dfs,
        #     'os': os,
        #     'val_fold': int(base_folder[-2]),
        #     'test_fold': int(base_folder[-1]),
        #     'vargrad_sum': s_d_norm.sum(),
        #     'vargrad_ct_sum': s_d_norm[..., 0].sum(),
        #     'vargrad_pt_sum': s_d_norm[..., 1].sum(),
        #     'vargrad_T_sum': s_d_norm[..., 2].sum(),
        #     'hu_corr_all': np.corrcoef(s_d_norm[..., 0].flatten(),
        #                                ct_img.flatten())[0, 1],
        #     'suv_corr_all': np.corrcoef(s_d_norm[..., 1].flatten(),
        #                                 pt_img.flatten())[0, 1],
        #     'hu_corr_raw': np.corrcoef(s_d_norm[..., 0].flatten(),
        #                                img[..., 0].flatten())[0, 1],
        #     'suv_corr_raw': np.corrcoef(s_d_norm[..., 1].flatten(),
        #                                 img[..., 1].flatten())[0, 1],
        #     'tumor_size': (tumor > 0).sum(),
        #     **get_area_info(s_d_norm, tumor, 'tumor_all'),
        #     'node_size': (node > 0).sum(),
        #     **get_area_info(s_d_norm, node, 'node_all'),
        #     **get_area_info(s_d_norm, 1 - tumor - node, 'outside_all'),
        #     **get_histogram_info(s_d_norm[..., 1], areas, area_names)
        # }

        # smooth_info = []
        # for quantile in [.95, .96, .97, .98, .99]:
        #     s_thres = np.quantile(smoothen_data, quantile)
        #     s_max_vargrad = smoothen_data.max()

        #     print('Normalizing smoothen interpret results...')
        #     s_data_normalized = ((smoothen_data - s_thres) /
        #                          (s_max_vargrad - s_thres)).clip(0, 1)
        #     smooth_info.append({
        #         **s_basic_info,
        #         'quantile': quantile,
        #         'vargrad_max': s_max_vargrad,
        #         'vargrad_threshold': s_thres,
        #         'vargrad_sum_selected': s_data_normalized.sum(),
        #         **get_info(s_data_normalized, ct_img, pt_img, tumor, node),
        #     })

        # print('Saving smoothen results...')
        # pd.DataFrame(smooth_info).to_csv(
        #     base_folder + f'/{center}/smoothen/{pid}.csv', index=False)

        print('Smoothening interpret results one more time...')
        smoothen_data = avg_filter(smoothen_data)
        print('Output shape:', smoothen_data.shape)

        # normalize original data
        s_d_min = smoothen_data.min()
        s_d_max = smoothen_data.max()

        s_d_norm = ((smoothen_data - s_d_min) / (s_d_max - s_d_min)).clip(0, 1)

        s_basic_info = {
            'pid': pid,
            'center': center,
            'dfs': dfs,
            'dfs_time': dfs_time,
            'os': os,
            'os_time': os_time,
            'val_fold': int(base_folder[-2]),
            'test_fold': int(base_folder[-1]),
            'vargrad_sum': s_d_norm.sum(),
            'vargrad_ct_sum': s_d_norm[..., 0].sum(),
            'vargrad_pt_sum': s_d_norm[..., 1].sum(),
            'vargrad_T_sum': s_d_norm[..., 2].sum(),
            'hu_corr_all': np.corrcoef(s_d_norm[..., 0].flatten(),
                                       ct_img.flatten())[0, 1],
            'suv_corr_all': np.corrcoef(s_d_norm[..., 1].flatten(),
                                        pt_img.flatten())[0, 1],
            'hu_corr_raw': np.corrcoef(s_d_norm[..., 0].flatten(),
                                       img[..., 0].flatten())[0, 1],
            'suv_corr_raw': np.corrcoef(s_d_norm[..., 1].flatten(),
                                        img[..., 1].flatten())[0, 1],
            'tumor_size': (tumor > 0).sum(),
            **get_area_info(s_d_norm, tumor, 'tumor_all'),
            'node_size': (node > 0).sum(),
            **get_area_info(s_d_norm, node, 'node_all'),
            **get_area_info(s_d_norm, 1 - tumor - node, 'outside_all'),
            **get_histogram_info(s_d_norm[..., 1], areas, area_names)
        }

        smooth_info = []
        for quantile in [.90, .93, .95, .97, .99]:
            s_thres = np.quantile(smoothen_data, quantile)
            s_max_vargrad = smoothen_data.max()

            print('Normalizing smoothen interpret results...')
            s_data_normalized = ((smoothen_data - s_thres) /
                                 (s_max_vargrad - s_thres)).clip(0, 1)
            smooth_info.append({
                **s_basic_info,
                'quantile': quantile,
                'vargrad_max': s_max_vargrad,
                'vargrad_threshold': s_thres,
                'vargrad_sum_selected': s_data_normalized.sum(),
                **get_info(s_data_normalized, ct_img, pt_img, tumor, node),
            })

        print('Saving smoothen results...')
        pd.DataFrame(smooth_info).to_csv(
            base_folder + f'/{center}/smoothen_v2/{pid}.csv', index=False)

    else:
        print('Index not found!! Exiting')
