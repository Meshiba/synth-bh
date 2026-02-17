import os
import errno
import argparse
import torch
from main_func import run_comparison
from utils import get_run_description, none_or_else
from utils_plot import plot


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.full_save_path:
        save_path = args.save_path + '/'
    else:
        save_path = args.save_path + '/' + get_run_description(args)
    if not os.path.exists(save_path + '/results/'):
        try:
            os.makedirs(save_path + '/results/', exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if args.plot:
        if not os.path.exists(save_path + '/plots/'):
            try:
                os.makedirs(save_path + '/plots/', exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    if args.p_test is None:
        args.p_test = args.p_cal
    if args.p_train is None:
        args.p_train = args.p_cal
    results = run_comparison(seed=args.seed, n_seeds=args.n_seeds,
                            n_cal=args.n_cal, p_cal=args.p_cal,
                            n_test=args.n_test, p_test=args.p_test,
                            n_train=args.n_train, p_train=args.p_train,
                            initial_cal=args.initial_labeled,
                            level=args.level,
                            epsilon=args.epsilon,
                            p_trim=args.p_trim,
                            model=args.model,
                            dataset=args.dataset,
                            dataset_version=args.dataset_version,
                            n_estimators=args.n_estimators,
                            args_dict=dict(vars(args)),
                            device=device,
                            save_path=save_path,
                            distribute=not args.no_distribute, slurm=not args.local,
                            )

    # save raw results
    results.to_pickle(save_path + '/results/results.pkl')
    # save plots
    if args.plot:
        for level_ in args.level:
            plot(results=results, level=level_, save_path=(save_path + f'/plots/level_{level_}/'), epsilon=args.epsilon)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_save_path', action='store_true', help='Do not add intermediate folder with run description.')
    parser.add_argument('--local', action='store_true', help='Run local experiments (not via SLURM).')
    parser.add_argument('--plot', action='store_true', help='Create plot figures, save and present them.')
    parser.add_argument('--no_distribute', action='store_true', help='Do not distribute. Run all seeds in the same '
                                                                     'process.')
    parser.add_argument('--save_path', type=str, default='./')
    parser.add_argument('--seed', type=int, default=42, help='Initial seed.')
    parser.add_argument('--n_seeds', type=int, default=100, help='Number of runs. Each run correspond to different '
                                                                 'seed.')

    parser.add_argument('--initial_labeled', type=int, default=0, help='Number of labeled samples.')
    parser.add_argument('--n_cal', type=int, default=1000, help='Number of calibration samples.')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test samples.')
    parser.add_argument('--p_trim', type=float, default=0.02, help='Proportion of scores to trim.')
    parser.add_argument('--p_cal', type=float, default=0.05, help='Proportion of outliers in the calibration-set.')
    parser.add_argument('--p_test', type=none_or_else, default=None, help='Proportion of outliers in the test-set (is None - same proportion as in the calibration set).')

    parser.add_argument('--level', type=none_or_else, default=['0.01'], nargs='+', help='Significant level to control.')
    parser.add_argument('--epsilon', type=float, default=0.01)
    # model
    parser.add_argument('--model', type=none_or_else, default=None, choices=['IF', None])
    parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples (same outliers '
                                                                  'proportion as in the calibration set unless otherwise specified).')
    parser.add_argument('--p_train', type=none_or_else, default=None, help='Proportion of outliers in the train-set.')
    parser.add_argument('--dataset', type=str, default='shuttle')
    parser.add_argument('--dataset_version', default=1)
    parser.add_argument('--n_estimators', type=int, default=100, help='IF - The number of base estimators in the '
                                                                      'ensemble')
    args = parser.parse_args()
    if args.n_train == 0 and args.model is not None:
        raise ValueError('n_train parameter must be > 0.')
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
