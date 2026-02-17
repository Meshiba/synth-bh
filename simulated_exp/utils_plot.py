import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
hue_order = ['BH_real', 'BH_synth',  'BH_pooled', 'SynthBH', 'BH_real+e']
palette4fig = {'BH_real': 'red', 'SynthBH': 'dodgerblue', 'BH_synth': 'darkorange', 'BH_real+e': 'limegreen',  'BH_pooled': 'darkorange'}
method2legend = {'BH_real': 'BH (real)', 'SynthBH': 'SynthBH', 'BH_synth': 'BH (synth)', 'BH_real+e': r'BH (real$+\varepsilon$)',  'BH_pooled': 'BH (pooled)',}
method2marker = {'SynthBH': 'o', 'BH_synth': 'D', 'BH_real': '^', 'BH_real+e': '*', 'BH_pooled': 'D',}
desc4paper_dict = {'n_real': r'$n$',
                   'n_synth': r'$N$',
                   'p_alt': r'$\rho$',
                   'p_null_synth': r'$\rho_{\text{synt}}^0$',
                   'p_alt_synth': r'$\rho_{\text{synt}}$',
                   'epsilon': r'$\varepsilon$',
                   'alpha': r'$\alpha$',
                   }


def get_run_description(p_alt=0.51, p_null_synth=None, p_alt_synth=0.51, n_real=50, n_synth=200, alpha=0.05, epsilon=0.02, 
                        n_runs=100, n_test=1000, p_test=0.05, exclude=[], x=None, x_min=None, x_max=None):
    run_name = '' if x is None else f'x_{x}'
    if x is not None and x_min is not None or x_max is not None:
        run_name += f'_{x_min}' if x_min is not None else '_'
        run_name += f'-{x_max}' if x_max is not None else '-'
    param_value_list = [('p_alt', p_alt), ('p_null_synth', p_null_synth), ('p_alt_synth', p_alt_synth),
                        ('n_real', n_real), ('n_synth', n_synth), ('n_test', n_test), ('p_test', p_test), ('alpha', alpha), ('epsilon', epsilon), ('n_runs', n_runs)]
    for param, value in param_value_list:
        if param not in exclude:
            if len(run_name):
                run_name += '_'
            run_name += param + '_' + str(value)
    return run_name


def desc4paper(x):
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    return x


def plot4paper(save_path, results, x=None, y='Power', hue='Method', methods2plot=None, alpha=0.05, epsilon=0.02, showmeans=True, ylim=None):
    font_size = 16
    plt.rc('legend', fontsize=font_size)
    if methods2plot is None:
        methods2plot = set(results['Method'].values)
    results = results.copy().reset_index()
    if x is None:
        raise ValueError('This function does not support x=None.')
    else:
        results = results[[x, hue, y]]
    results_ = results.loc[results['Method'].isin(methods2plot)]
    results_ = results_.replace([np.inf, -np.inf], np.nan)
    results_ = results_.dropna(inplace=False)
    if results_.empty:
        print('Results are empty after dropping None and inf values --- skipping plot generation...')
        return
    fig = plt.figure(figsize=([5,3]))
    ax = fig.add_subplot(111)
    hue_order_curr = [m for m in hue_order if m in methods2plot]
    for m in methods2plot:
        if not m in hue_order_curr:
            hue_order_curr.append(m)
    default_palette = sns.color_palette('PuBuGn', len(hue_order_curr))
    curr_palette = {}
    for i, m in enumerate(hue_order_curr):
        if m in palette4fig.keys():
            curr_palette[m] = palette4fig[m]
        else:
            curr_palette[m] = default_palette[i]
    sns.lineplot(data=results_, x=x, y=y, hue=hue, ax=ax, palette=curr_palette, hue_order=hue_order_curr)
    means = results_.groupby([hue, x])[y].mean().reset_index()

    for group in hue_order_curr:
        group_means = means[means[hue] == group]
        plt.scatter(
            group_means[x],
            group_means[y],
            marker=method2marker[group],
            s=70,              # marker size
            label=None,        # don't duplicate legend
            zorder=5,           # place markers above shaded area
            # edgecolor='black',
            color=curr_palette[group],
        )

    plt.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xlabel(desc4paper(x), fontsize=font_size)
    ax.set_ylabel(y, fontsize=font_size)
    if y == 'FDR':
        if x == 'alpha':
            x_values = np.sort(results_[x].values)
            ax.plot(x_values, x_values, linestyle='dashed', color='black', label=r'$\alpha$')
            ax.plot(x_values, [x_ + epsilon for x_ in x_values], linestyle='dashed', color='darkgray', label=r'$\alpha + \varepsilon$')
        elif x == 'epsilon':
            x_values = np.sort(results_[x].values)
            ax.axhline(alpha, linestyle='dashed', color='black', label=r'$\alpha$')
            ax.plot(x_values, [x_ + alpha for x_ in x_values], linestyle='dashed', color='darkgray', label=r'$\alpha + \varepsilon$')
        else:
            ax.axhline(alpha, linestyle='dashed', color='black', label=r'$\alpha$')
            ax.axhline(alpha+epsilon, linestyle='dashed', color='darkgray', label=r'$\alpha + \varepsilon$')
    fig.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)
    handles, labels = ax.get_legend_handles_labels()
    labels = [method2legend[x] if x in method2legend.keys() else x for x in labels]
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 0.5))
    os.makedirs(save_path + '/plots/', exist_ok=True)
    plt.savefig(save_path + '/plots/' + y + '.pdf', bbox_inches="tight")
    plt.savefig(save_path + '/plots/' + y + '.png', bbox_inches="tight")
    ax.get_legend().remove()
    plt.savefig(save_path + '/plots/' + y + '_no_legend.pdf', bbox_inches='tight')
    plt.savefig(save_path + '/plots/' + y + '_no_legend.png', bbox_inches='tight')
    plt.close()


def load_results(results_dir):
    results = pd.DataFrame({})
    files = glob.glob(os.path.join(results_dir, '*/results/results.pkl'))
    for f in files:
        curr_results = pd.read_pickle(f)
        results = pd.concat([results, curr_results], ignore_index=True)
    return results


def filter_and_plot(save_path, results, x, y='Power', methods2plot=None, alpha=0.05, epsilon=0.02, x_min=None, x_max=None, x_filter=None, ylim=None, **kwargs):
    curr_results = results.copy()
    for k, v in kwargs.items():
        if isinstance(v, list):
            curr_results = curr_results[curr_results[k].astype(str).isin(v)]
        else:
            curr_results = curr_results[curr_results[k].astype(str) == str(v)]
    if x != 'alpha':
        curr_results = curr_results[curr_results['alpha'].astype(str) == str(alpha)]
    if x != 'epsilon':
        curr_results = curr_results[curr_results['epsilon'].astype(str) == str(epsilon)]
    if not 'p_null_synth' in kwargs.keys() and x != 'p_null_synth':
        curr_results = curr_results[curr_results['p_null_synth'].astype(float) > 0.5]
    if x_max is not None:
        x_ = x if x_filter is None else x_filter
        curr_results = curr_results[curr_results[x_] <= x_max]
    if x_min is not None:
        x_ = x if x_filter is None else x_filter
        curr_results = curr_results[curr_results[x_] >= x_min]
    if curr_results.empty:
        print('Results data frame is empty after filtering... skip')
    # create dir for run name
    exclude=[x]
    if x_filter is not None:
        exclude.append(x_filter)
    if not 'n_runs' in kwargs.keys():
        exclude.append('n_runs')
    params = {'alpha': curr_results['alpha'].values[0],
              'epsilon': curr_results['epsilon'].values[0],
              **kwargs
              }
    if 'n_real' not in params.keys():
        params['n_real'] = curr_results['n_real'].values[0]
    if 'n_synth' not in params.keys():
        params['n_synth'] = curr_results['n_synth'].values[0]
    curr_save_path = save_path + '/' + get_run_description(**params,
                                                           exclude=exclude,
                                                           x=x,
                                                           x_min=x_min, x_max=x_max,
                                                           )
    plot4paper(curr_save_path, curr_results, x=x, y=y, methods2plot=methods2plot, alpha=alpha, epsilon=epsilon, ylim=ylim)
    plt.close()


def load_and_plot(save_path, results_dir, x, y='Power', methods2plot=None, alpha=0.05, epsilon=0.02, x_min=None, x_max=None, x_filter=None, ylim=None, **kwargs):
    results = load_results(results_dir)
    return filter_and_plot(save_path, results, x, y=y, methods2plot=methods2plot,
                           alpha=alpha, epsilon=epsilon, x_min=x_min, x_max=x_max, x_filter=x_filter, ylim=ylim,
                           **kwargs)
