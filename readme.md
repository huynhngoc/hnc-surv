# Deep learning for survival analysis of head and neck cancer

## Run the experiments locally
`python experiment_survival_local.py [path_to_config_file] [path_to_log_file] --epochs [epoch_num]`

For example, the following script will run the experiment from DFS_CT_PET_tmp_model_f01234 for 2 epochs and save the data into surv_perf
`python experiment_survival_local.py config/survival_img_CT_PET/DFS_CT_PET_tmp_model_f01234.json ../surv_perf --epochs 2`
