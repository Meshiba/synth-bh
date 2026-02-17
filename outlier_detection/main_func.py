import shutil
import tempfile
import time
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import torch
from utils import set_seed, get_model, generate_all_data, generate_score_data, get_latent_rep, draw_number_of_outliers_inliers, flatten_array
from experiments.utils import create_command, run_command, get_params_for_one_seed_command, check_for_job_state
from algo import get_rejections_indices, analyze_performance, get_calibration_set


def run_comparison(seed=42, n_seeds=100, initial_cal=0,
                   n_cal=10000, p_cal=0.05,
                   n_test=10000, p_test=0.05,
                   n_train=10000, p_train=0.05,
                   level=0.1, epsilon=0.01, p_trim=0.1,
                   model=None,
                   dataset="shuttle", dataset_version=1,
                   n_estimators=100, args_dict=None,
                   device="cpu",
                   distribute=True, save_path=None, slurm=True, 
                   ):
    if not distribute or n_seeds == 1:
        return run_one_comparison(seed=seed, n_seeds=n_seeds, initial_cal=initial_cal,
                   n_cal=n_cal, p_cal=p_cal,
                   n_test=n_test, p_test=p_test,
                   n_train=n_train, p_train=p_train,
                   level=level, epsilon=epsilon, p_trim=p_trim,
                   model=model,
                   dataset=dataset, dataset_version=dataset_version,
                   n_estimators=n_estimators, args_dict=args_dict,
                   device=device,
                   save_path=save_path,
                    )
    results = pd.DataFrame({})
    all_process = []
    all_commands = []
    all_result_paths = []
    set_seed(seed)
    # create tmp results dir in save_path
    tmp_files_path = save_path + '/tmp_results/'
    try:
        os.makedirs(tmp_files_path)
    except:
        pass
    seed_list = random.sample(range(1, 999999), n_seeds)
    for seed_ in tqdm(seed_list):
        # create tmp result dir for this specific run
        tmp_dir_path = tempfile.mkdtemp(dir=tmp_files_path)
        all_result_paths.append(tmp_dir_path)
        params, flag_params = get_params_for_one_seed_command(seed=seed_, args_dict=args_dict)
        command = create_command(tmp_dir_path + '/', params, flag_params)
        if slurm:
            command = f"bash ./create_tmp_empty.sh \"{command}\""
        process = run_command(command, read_out=True)
        all_process.append(process)
        all_commands.append(command)
    # wait for all processes to finish
    if slurm:
        job_ids = []
        for process in all_process:
            output, _ = process.communicate()
            job_id = str(output).split(' ')[-1].strip()
            job_ids.append(job_id)
        all_job_ids = job_ids.copy()
        while len(job_ids):
            state = check_for_job_state(job_ids[0])
            if state == 'complete':
                del job_ids[0]
            else:
                time.sleep(30)
    while len(all_process):
        all_process[0].communicate()
        # load results
        if os.path.isfile(all_result_paths[0] + '/results/results.pkl'):
            curr_result = pd.read_pickle(all_result_paths[0] + '/results/results.pkl')
            results = pd.concat([results, curr_result])
            # delete tmp dir results for this run
            shutil.rmtree(all_result_paths[0])
            del all_process[0]
            del all_commands[0]
            del all_result_paths[0]
            if slurm:
                del all_job_ids[0]
        else:
            shutil.rmtree(tmp_files_path, ignore_errors=True)
            job_desc = f'Failed job id {all_job_ids[0]}\n' if slurm else ''
            raise ValueError(f'{job_desc}Deleting tmp results dir...\nThe following run falied to save results file:\n{all_commands[0]}')
    # delete tmp results folder
    shutil.rmtree(tmp_files_path, ignore_errors=True)
    return results


def run_one_comparison(seed=42, n_seeds=100, initial_cal=0,
                       n_cal=10000, p_cal=0.05,
                       n_test=10000, p_test=0.05,
                       n_train=10000, p_train=0.05,
                       level=0.1, epsilon=0.01, p_trim=0.1,
                       model=None,
                       dataset="shuttle", dataset_version=1,
                       n_estimators=100, args_dict=None,
                       device="cpu",
                       save_path=None,
                       ):
    results = pd.DataFrame({})
    methods = ['BH_synth', 'BH_pooled']
    # update params in args_dict
    if initial_cal >= 0:
        methods.append('BH_real')
        methods.append('BH_real+e')
        methods.append('SynthBH')
    set_seed(seed)
    if n_seeds == 1:
        seed_list = [seed]
    else:
        seed_list = random.sample(range(1, 999999), n_seeds)
    for seed_ in tqdm(seed_list):
        set_seed(seed_)
        # random hypotheses - draw number of samples
        cal_n_inliers, cal_n_outliers, test_n_inliers, test_n_outliers, data_n_inliers, data_n_outliers, \
        train_n_inliers, train_n_outliers = draw_number_of_outliers_inliers(n_cal, p_cal, n_test, p_test,
                                                           n_train, p_train, initial_cal, random=False)

        # generate data
        calib_dataset, test_dataset, train_dataset, initial_calib = \
            generate_all_data(cal_n_inliers=cal_n_inliers, cal_n_outliers=cal_n_outliers,
                            test_n_inliers=test_n_inliers, test_n_outliers=test_n_outliers,
                            initial_cal=initial_cal,
                            train_n_inliers=train_n_inliers, train_n_outliers=train_n_outliers,
                            dataset=dataset,
                            dataset_version=dataset_version)
        # model
        model_ = get_model(model, n_estimators, device=device)
        # train model
        train_set, _ = get_latent_rep(train_dataset)
        if len(train_set.shape) == 1 or train_set.shape[1] == 1:
            train_set = train_set.reshape((-1,1))
        model_.fit(train_set)

        # transform all data to scores
        calib_set, calib_y = get_latent_rep(calib_dataset)
        test_set, test_y = get_latent_rep(test_dataset)
        if initial_cal > 0:
            initial_calib_set, _ = get_latent_rep(initial_calib)
        else:
            initial_calib_set = np.array([])
        if len(calib_set.shape) == 1 or calib_set.shape[1] == 1:
            calib_set = calib_set.reshape((-1,1))
            test_set = test_set.reshape((-1,1))
            if len(initial_calib_set):
                initial_calib_set = initial_calib_set.reshape((-1,1))

        if not dataset.startswith('scores_'):
            # get scores from model
            calib_set = -1 * model_.decision_function(calib_set)
            test_set = -1 * model_.decision_function(test_set)
            if initial_cal > 0:
                initial_calib_set = -1 * model_.decision_function(initial_calib_set)

        if torch.is_tensor(calib_set):
            calib_set, test_set = calib_set.numpy(), test_set.numpy()
            if initial_cal > 0:
                initial_calib_set = initial_calib_set.numpy()

        noise_level = (10**-15)
        calib_set = calib_set.reshape((-1,))
        noise = np.random.normal(0, 1, size=calib_set.shape) * noise_level
        calib_set += noise
        test_set = test_set.reshape((-1,))
        noise = np.random.normal(0, 1, size=test_set.shape) * noise_level
        test_set += noise
        # shuffle test set
        randomized = np.random.permutation(len(test_set))
        test_set, test_y = test_set[randomized], test_y[randomized]
        if len(initial_calib_set):
            initial_calib_set = initial_calib_set.reshape((-1,))
            noise = np.random.normal(0, 1, size=initial_calib_set.shape) * noise_level
            initial_calib_set += noise

        for curr_method in methods:
            curr_calib_set, curr_calib_y = get_calibration_set(curr_method, initial_cal,
                                                                initial_calib_set, calib_set, calib_y,
                                                                p_trim)
            if isinstance(level, list):
                levels = level
            else:
                levels = [level]
            for curr_level in levels:
                curr_level = float(curr_level)

                if curr_calib_set is not None and len(curr_calib_set):
                    curr_rejections, curr_threshold = get_rejections_indices(curr_calib_set, test_set, level=curr_level,
                                                                             epsilon=epsilon, method=curr_method)
                    curr_power, curr_fdr = analyze_performance(curr_rejections, test_y)
                else:
                    curr_calib_set, curr_calib_y = np.array([]), np.array([])
                    curr_rejections, curr_threshold = [], np.inf
                    curr_power = 0
                n_rejections = len(flatten_array(curr_rejections)) if curr_rejections is not None and len(curr_rejections) > 0 else 0
                curr_results_dict = {'Type': curr_method, 'Power': curr_power,
                                     'Threshold': curr_threshold,
                                     'FDR': curr_fdr,
                                     'Rejections': n_rejections, 'Seed': seed_,
                                     'level': curr_level, 'epsilon': epsilon}
                curr_result = pd.DataFrame(curr_results_dict, index=[0])
                results = pd.concat([results, curr_result])

    # add all args to results
    if args_dict is not None:
        for k,v in args_dict.items():
            if isinstance(v, list):
                continue
            results[k] = v
    return results
