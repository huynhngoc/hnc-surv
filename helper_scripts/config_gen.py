import json
import os

filenames = [f for f in os.listdir(
    'config/outcome_img_CT_PET') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/outcome_img_CT_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        config['dataset_params']['config']['y_name'] = 'OS_surv'
        config['dataset_params']['config']['preprocessors'].append({
            "class_name": "DummyRiskScoreConverter",
            "config": {
                "vmin": 0,
                "vmax": 100
            }
        })
        config['model_params']['loss'] = {
            "class_name": "SurvLoss",
            "config": {
                "loss_config": {
                    "class_name": "MeanSquaredError"
                }
            }
        }
        config['model_params']['metrics'] = [
            {
                "class_name": "SurvMetric",
                "config": {
                    "name": "mse_customized",
                    "metric_config": {
                        "class_name": "MeanSquaredError"
                    }
                }
            }
        ]
        # config['architecture']['layers'][-1]['config']['units'] = 10
    new_fn = fn.replace('3d_eff_b1m16', 'dummy')
    with open(f'config/survival_img_CT_PET/{new_fn}', 'w') as f:
        json.dump(config, f)


filenames = [f for f in os.listdir(
    'config/outcome_img_CT_PET') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/outcome_img_CT_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        config['dataset_params']['config']['y_name'] = 'OS_surv'
        config['dataset_params']['config']['preprocessors'].append({
            "class_name": "MakeSurvArray",
            "config": {
                "breaks": [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
            }
        })
        config['model_params']['loss'] = {
            "class_name": "NegativeLogLikelihood",
            "config": {
                "n_intervals": 10
            }
        }
        config['model_params']['metrics'] = []
        config['architecture']['layers'][-1]['config']['units'] = 10
    new_fn = fn.replace('3d_eff_b1m16', 'loglikelihood')
    with open(f'config/survival_img_CT_PET/{new_fn}', 'w') as f:
        json.dump(config, f)


# ======== generating the PET + GTVp config
# filename = 'config/survival_img_CT_PET_test_local/DFS_PET_tmp_model_f01234_loglikelihood.json'
# with open(filename, 'r') as f:
#     config = json.load(f)
# # find the channel selector preprocess
# for prep in config['dataset_params']['config']['preprocessors']:
#     if prep['class_name'] == 'ChannelSelector':
#         prep['config']['channel'] = [1,2]
# config['input_params']['shape'][-1] = 2
# config['architecture']['layers'][0]['inputs'] = ['input_0', 'PET_masked']
# config['architecture']['layers'].insert(0, {
#                 "name": "PET_masked",
#                 "class_name": "DepthwiseMultiply",
#                 "config": {
#                     "channels": [
#                         0,
#                         1
#                     ]
#                 },
#                 "inputs": [
#                     "input_0"
#                 ]
#             })
# with open('config/survival_img_CT_PET_test_local/DFS_PET_tumor_tmp_model_f01234_loglikelihood_test.json', 'w') as f:
#     json.dump(config, f)


# from PET only to PET+GTVp
filenames = [f for f in os.listdir(
    'config/survival_img_PET') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/survival_img_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        for prep in config['dataset_params']['config']['preprocessors']:
            if prep['class_name'] == 'ChannelSelector':
                prep['config']['channel'] = [1, 2]
        config['input_params']['shape'][-1] = 2
        config['architecture']['layers'][0]['inputs'] = [
            'input_0', 'PET_masked']
        config['architecture']['layers'].insert(0, {
            "name": "PET_masked",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    0,
                    1
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
    new_fn = fn.replace('PET', 'PET_tumor')
    with open(f'config/survival_img_PET_tumor/{new_fn}', 'w') as f:
        json.dump(config, f)


# from PET only to PET+GTVp + GTVn
filenames = [f for f in os.listdir(
    'config/survival_img_PET') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/survival_img_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        for prep in config['dataset_params']['config']['preprocessors']:
            if prep['class_name'] == 'ChannelSelector':
                prep['config']['channel'] = [1, 2, 3]
        config['input_params']['shape'][-1] = 3
        config['architecture']['layers'][0]['inputs'] = [
            'input_0', 'PET_tumor', 'PET_node']
        config['architecture']['layers'].insert(0, {
            "name": "PET_node",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    0,
                    2
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
        config['architecture']['layers'].insert(0, {
            "name": "PET_tumor",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    0,
                    1
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
    new_fn = fn.replace('PET', 'PET_tumor_node')
    with open(f'config/survival_img_PET_tumor_node/{new_fn}', 'w') as f:
        json.dump(config, f)


# from PET only to CT_only
filenames = [f for f in os.listdir(
    'config/survival_img_PET') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/survival_img_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        for prep in config['dataset_params']['config']['preprocessors']:
            if prep['class_name'] == 'ChannelSelector':
                prep['config']['channel'] = [0]
            if prep['class_name'] == 'ImageNormalizerPreprocessor':
                prep['config'] = {
                    "vmin": [
                        -100
                    ],
                    "vmax": [
                        100
                    ]
                }
        config['dataset_params']['config']['preprocessors'].insert(1, {
            "class_name": "HounsfieldWindowingPreprocessor",
            "config": {
                "window_center": 70,
                "window_width": 200,
                "channel": 0
            }
        })
        config['input_params']['shape'][-1] = 1

    new_fn = fn.replace('PET', 'CT')
    with open(f'config/survival_img_CT/{new_fn}', 'w') as f:
        json.dump(config, f)


# from CT only to CT+GTVp
filenames = [f for f in os.listdir(
    'config/survival_img_CT') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/survival_img_CT/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        for prep in config['dataset_params']['config']['preprocessors']:
            if prep['class_name'] == 'ChannelSelector':
                prep['config']['channel'] = [0, 2]
        config['input_params']['shape'][-1] = 2
        config['architecture']['layers'][0]['inputs'] = [
            'input_0', 'CT_masked']
        config['architecture']['layers'].insert(0, {
            "name": "CT_masked",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    0,
                    1
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
    new_fn = fn.replace('CT', 'CT_tumor')
    with open(f'config/survival_img_CT_tumor/{new_fn}', 'w') as f:
        json.dump(config, f)


# from CT only to CT+GTVp + GTVn
filenames = [f for f in os.listdir(
    'config/survival_img_CT') if f.startswith('OS')]

for fn in filenames:
    with open(f'config/survival_img_CT/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        for prep in config['dataset_params']['config']['preprocessors']:
            if prep['class_name'] == 'ChannelSelector':
                prep['config']['channel'] = [0, 2, 3]
        config['input_params']['shape'][-1] = 3
        config['architecture']['layers'][0]['inputs'] = [
            'input_0', 'CT_tumor', 'CT_node']
        config['architecture']['layers'].insert(0, {
            "name": "CT_node",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    0,
                    2
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
        config['architecture']['layers'].insert(0, {
            "name": "CT_tumor",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    0,
                    1
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
    new_fn = fn.replace('CT', 'CT_tumor_node')
    with open(f'config/survival_img_CT_tumor_node/{new_fn}', 'w') as f:
        json.dump(config, f)


# from PET/CT to PET/CT+GTVp
filenames = [f for f in os.listdir(
    'config/survival_img_CT_PET') if f.startswith('OS_CT_PET_log')]

for fn in filenames:
    with open(f'config/survival_img_CT_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        for prep in config['dataset_params']['config']['preprocessors']:
            if prep['class_name'] == 'ChannelRemoval':
                prep['config']['channel'] = [3]
        config['input_params']['shape'][-1] = 3
        config['architecture']['layers'][0]['inputs'] = [
            'input_0', 'CT_masked', 'PET_masked']
        config['architecture']['layers'].insert(0, {
            "name": "PET_masked",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    1,
                    2
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
        config['architecture']['layers'].insert(0, {
            "name": "CT_masked",
            "class_name": "DepthwiseMultiply",
            "config": {
                    "channels": [
                        0,
                        2
                    ]
            },
            "inputs": [
                "input_0"
            ]
        })
    new_fn = fn.replace('CT_PET', 'CT_PET_tumor')
    with open(f'config/survival_img_CT_PET_tumor/{new_fn}', 'w') as f:
        json.dump(config, f)


# from PET/CT to PET/CT+GTVp+GTVn
filenames = [f for f in os.listdir(
    'config/survival_img_CT_PET') if f.startswith('OS_CT_PET_log')]

for fn in filenames:
    with open(f'config/survival_img_CT_PET/{fn}', 'r') as f:
        config = json.load(f)
        # config['dataset_params']['config']['filename'] = '/mnt/project/ngoc/hn_surv/datasets/outcome_ous.h5'
        config['dataset_params']['config']['preprocessors'].pop(0)
        config['input_params']['shape'][-1] = 4
        config['architecture']['layers'][0]['inputs'] = [
            'input_0', 'CT_tumor', 'PET_tumor', 'CT_node', 'PET_node']
        config['architecture']['layers'].insert(0, {
            "name": "PET_node",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    1,
                    3
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
        config['architecture']['layers'].insert(0, {
            "name": "CT_node",
            "class_name": "DepthwiseMultiply",
            "config": {
                    "channels": [
                        0,
                        3
                    ]
            },
            "inputs": [
                "input_0"
            ]
        })
        config['architecture']['layers'].insert(0, {
            "name": "PET_tumor",
            "class_name": "DepthwiseMultiply",
            "config": {
                "channels": [
                    1,
                    2
                ]
            },
            "inputs": [
                "input_0"
            ]
        })
        config['architecture']['layers'].insert(0, {
            "name": "CT_tumor",
            "class_name": "DepthwiseMultiply",
            "config": {
                    "channels": [
                        0,
                        2
                    ]
            },
            "inputs": [
                "input_0"
            ]
        })
    new_fn = fn.replace('CT_PET', 'CT_PET_tumor_node')
    with open(f'config/survival_img_CT_PET_tumor_node/{new_fn}', 'w') as f:
        json.dump(config, f)
