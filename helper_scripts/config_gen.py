import json
import os

filenames = [f for f in os.listdir('config/outcome_img_CT_PET') if f.startswith('OS')]

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
                    "config":{
                        "loss_config": {
                            "class_name": "MeanSquaredError"
                        }
                    }
                }
        config['model_params']['metrics'] = [
            {
                "class_name": "SurvMetric",
                "config":{
                    "name":"mse_customized",
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
