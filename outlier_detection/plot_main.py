import argparse
import os
from utils_plot import plot_xy
import pandas as pd
import numpy as np


def get_data(file_path):
    results = pd.read_pickle(file_path)
    return results


def round_to_0_005(x):
    if pd.isna(x):  # leave NaN as-is
        return x
    x = round(x, 3)
    return x



def plot_figures(results, x, y, save_path=None, filter_={}, file_desc=''):
    for k,v in filter_.items():
        results = results[results[k].astype(str) == str(v)]
    if results.empty:
        print('Results dataframe is empty... exit')
        return
    # plot for every level
    if x != 'level' and x != 'emp_fdr':
        methods2plot = ['SynthBH', 'BH_real', 'BH_pooled', 'BH_real+e']
        results = results.loc[results['Type'].isin(methods2plot)]
        all_levels = np.unique(results['level'].values)
        for level in all_levels:
            curr_results = results.copy()
            curr_results = curr_results[curr_results['level'] == level]
            plot_xy(curr_results, x=x, y=y, level=None, save_path=os.path.join(save_path, f'level_{level}'), extract_data=True, file_desc=file_desc)
    else:
        if x == 'emp_fdr':
            args_to_group = ['level', 'dataset', 'Type']
            ####################################################################
            results['emp_fdr'] = (
                results.groupby(args_to_group)['FDR']
                .transform('mean')
                .apply(round_to_0_005)
            )

            alpha_levels = sorted(results['level'].unique())
            methods = results['Type'].unique()
            seen = {method: set() for method in methods}  # track seen mean_emp_coverage per method
            rows_to_keep = []
            for alpha in alpha_levels:
                for method in methods:
                    # all rows for this alpha & method
                    mask = (results['level'] == alpha) & (results['Type'] == method)
                    if not mask.any():
                        print(method, alpha)
                        continue
                    mean_val = results.loc[mask, 'emp_fdr'].iloc[0]  # all 100 rows have the same mean
                    if mean_val is not None and mean_val not in seen[method]:
                        # keep all rows for this alpha & method
                        rows_to_keep.append(results[mask])
                        # mark as seen
                        seen[method].add(mean_val)

            # Combine kept rows
            results = pd.concat(rows_to_keep).reset_index(drop=True)
            methods2plot = ['SynthBH', 'BH_real']
            results = results.loc[results['Type'].isin(methods2plot)]
            #################################################################
        plot_xy(results, x=x, y=y, level=None, save_path=save_path, extract_data=True, file_desc=file_desc)
        plot_xy(results, x=x, y=y, level=None, save_path=save_path, extract_data=True, file_desc='line_' + file_desc, box=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, nargs='+', default='./results/')
    parser.add_argument('--plot_dir', type=str, default='./plots/')
    parser.add_argument('--x', type=str, default='dataset')
    parser.add_argument('--y', type=str, nargs='+', default=None)
    parser.add_argument('--filter_k', type=str, nargs='+', default=[])
    parser.add_argument('--filter_v', nargs='+', default=[])
    parser.add_argument('--file_desc', type=str, default='', help='Suffix to append to the filename for the plot output.')
    args = parser.parse_args()
    if len(args.filter_k) != len(args.filter_v):
        raise ValueError('The length of filter keys and values must be the same.')
    return args


def main(args):
    filter_dict = {}
    for i in range(len(args.filter_k)):
        filter_dict[args.filter_k[i]] = args.filter_v[i]
    all_results = pd.DataFrame({})
    if not isinstance(args.result_dir, list):
        args.result_dir = [args.result_dir]
    result_files = []
    for res_dir in args.result_dir:
        # result_files.extend([os.path.join(res_dir, f) for f in os.listdir(res_dir)])
        for dirpath, dirnames, filenames in os.walk(res_dir):
            if os.path.basename(dirpath) == "results":
                pkl_path = os.path.join(dirpath, "results.pkl")
                if os.path.isfile(pkl_path):
                    result_files.append(pkl_path)
    for f in result_files:
        results = get_data(f)
        all_results = pd.concat([all_results, results])
    plot_figures(all_results, x=args.x, y=args.y, save_path=args.plot_dir, filter_=filter_dict,
                    file_desc=args.file_desc)


if __name__ == "__main__":
    args = get_args()
    main(args)

