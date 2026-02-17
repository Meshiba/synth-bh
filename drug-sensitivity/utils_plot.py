import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
hue_order = ['BH_real', 'BH_synth', 'BH_pooled', 'SynthBH', 'BH_real+e']
palette4fig = {'BH_real': 'red', 'SynthBH': 'dodgerblue', 'BH_pooled': 'darkorange', 'BH_synth': 'darkorange', 'BH_real+e': 'limegreen'}
method2legend = {'BH_real': 'BH (real)', 'SynthBH': 'SynthBH', 'BH_synth': 'BH (only-synth)', 'BH_real+e': r'BH (real$+\varepsilon$)', 'BH_pooled': 'BH (pooled)',}
method2marker = {'SynthBH': 'o', 'BH_synth': 'D', 'BH_real': '^', 'BH_real+e': '*', 'BH_pooled': 'D',}
desc4paper_dict = {
                   'epsilon': r'$\varepsilon$',
                   'alpha': r'$\alpha$',
                   'n_rejections': r'Number of Rejections',
                   'gt_score': r'Ground Truth Score',
                   }


def desc4paper(x):
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    return x


def plot4paper(save_path, results, x=None, y='Power', hue='Method', methods2plot=None, alpha=0.05, epsilon=0.02, showmeans=True, ylim=None, remove_x=False):
    font_size = 18
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
    fig = plt.figure(figsize=([3,3.5]))
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
    sns.boxplot(data=results_, x=x, y=y, hue=hue, ax=ax, palette=curr_palette, hue_order=hue_order_curr)
    if showmeans:
        hue_levels = hue_order_curr
        if x is None:
            category_levels = [0]
        else:
            category_levels = sorted(list(results[x].unique()))
        if len(hue_levels) == 5:
            offsets = np.linspace(-0.32, 0.32, len(hue_levels))  # Adjust for dodge
        else:
            offsets = np.linspace(-0.3, 0.3, len(hue_levels))  # Adjust for dodge
        if len(results[hue].unique()) == 1:
            offsets = [0,] * len(hue_levels)
        for i, category in enumerate(category_levels):
            for j, hue_ in enumerate(hue_levels):
                if x is None:
                    subset = results[(results[hue] == hue_)]
                else:
                    subset = results[(results[x] == category) & (results[hue] == hue_)]
                if not subset.empty:
                    mean_value = subset[y].mean()  # Compute mean
                    x_position = i + offsets[j]  # Adjust x position for dodge
                    ax.scatter(x_position, mean_value, color=palette4fig[hue_], edgecolor='black', s=70, zorder=3, marker=method2marker[hue_])


    plt.tick_params(axis='both', which='major', labelsize=font_size)
    ax.set_xlabel(desc4paper(x), fontsize=font_size)
    ax.set_ylabel(desc4paper(y), fontsize=font_size)
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
    if remove_x:
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.set_xlim(-0.7,0.7)
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

