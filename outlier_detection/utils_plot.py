import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
method2legend = {'BH_synth': 'BH_synth', 'BH_pooled': 'BH_pooled',
                 'BH_real': 'BH_real', 'SynthBH': 'SynthBH', 'BH_real+e': 'Guardrail'}
palette4fig = {'BH_synth': 'orange', 'BH_real': 'red', 'SynthBH': 'dodgerblue', 'BH_real+e': 'limegreen',
               'BH_pooled': 'orange'}
markers4fig = {'BH_synth': 'D', 'BH_real': '^', 'SynthBH': 'o', 'BH_real+e': '*', 'BH_pooled': 'D'}
hue_order_g = ['BH_real', 'BH_synth', 'BH_pooled', 'SynthBH', 'BH_real+e']
hue_order_g_legend = [method2legend[m] if m in method2legend.keys() else m for m in hue_order_g]


def desc4paper(x):
    desc4paper_dict = {
                       'dataset': 'Dataset',
                       'p_trim': r'$\rho$',
                       'level': r'$\alpha$',
                       'initial_labeled': r'$n$',
                       'epsilon': r'$\varepsilon$',
                       'emp_fdr': r'$\widehat{FDR}$',
                       'FDR': 'FDP',
                       'Power': 'Empirical Power',
                       }
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    else:
        return x


def image_dataset2labels(s):
    try:
        ood_dataset_name = s.split('_')[2]
        return ood_dataset_name[0].upper() + ood_dataset_name[1:]
    except:
        return s


def plot_xy(results, level, save_path=None, x='dataset', y=[], hue='Type', extract_data=False, file_desc='', epsilon=0.01, showmeans=True, box=True):
    global hue_order_g
    curr_hue_order_g = [m for m in hue_order_g if m in results['Type'].unique()]
    font_size = 16
    if x == 'emp_fdr':
        font_size_labels = 16  #20
        font_size_legend = 16  # 20
    else:
        font_size_labels = 20
        font_size_legend = 20
    if extract_data:
        level = results['level'].values[0]
        epsilon = results['epsilon'].values[0]
        print(f'Parameters for plot: alpha={level}, epsilon={epsilon}')
    # filter according to args
    if x != 'level' and x != 'emp_fdr' and x != 'FDR':
        results = results[results['level'].astype(str) == str(level)]
        level = float(level)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    if y is None or len(y) == 0:
        y = ['Power', 'FDR']
    all_results = results
    for y_ in y:
        if x == 'emp_fdr':
            fig = plt.figure(figsize=([4, 3]))
        else:
            fig = plt.figure(figsize=([6, 4]))
        ax = fig.add_subplot(111)
        results = all_results
        if x == 'FDR':
            print(len(results))
            sns.scatterplot(results, x=x, hue=hue, style=hue, y=y_, palette=palette4fig, ax=ax, hue_order=curr_hue_order_g, s=30)
            showmeans = False
        else:
            if box:
                sns.boxplot(results, x=x, hue=hue, y=y_, palette=palette4fig, ax=ax, hue_order=curr_hue_order_g)
            else:
                sns.lineplot(results, x=x, hue=hue, y=y_, palette=palette4fig, ax=ax, hue_order=curr_hue_order_g, marker="o")
                showmeans = False
        if y_ == 'FDR' and (x != 'level' and x != 'emp_fdr'):
            ax.axhline(level, color='black', linestyle='dashed', label=r'$\alpha$')
            ax.axhline(level + epsilon, color='gray', linestyle='dashed', label=r'$\alpha + \varepsilon$')
        elif y_ == 'FDR' and x == 'level':
            alphas = sorted(list(set(results[x])))
            width = 0.45
            for i, a in enumerate(alphas):
                if i == 0:
                    ax.hlines(y=a, xmin=i - width, xmax=i + width, color='black', linestyle='dashed', label=r'$\alpha$')
                    ax.hlines(y=a + epsilon, xmin=i - width, xmax=i + width, color='gray', linestyle='dashed', label=r'$\alpha+\varepsilon$')
                else:
                    ax.hlines(y=a, xmin=i - width, xmax=i + width, color='black', linestyle='dashed')
                    ax.hlines(y=a + epsilon, xmin=i - width, xmax=i + width, color='gray', linestyle='dashed')
        if showmeans:
            hue_levels = curr_hue_order_g
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
                        mean_value = subset[y_].mean()  # Compute mean
                        x_position = i + offsets[j]  # Adjust x position for dodge
                        ax.scatter(x_position, mean_value, color=palette4fig[hue_], edgecolor='black', s=70, zorder=3, marker=markers4fig[hue_])

        plt.rc('legend', fontsize=font_size_legend)
        if x != 'emp_fdr':
            plt.xticks(rotation=45)
        else:
            plt.margins(x=0.015)      # small breathing room
        locs, labels = plt.xticks()
        if x == 'emp_fdr':
            if float(labels[0].get_text().replace('−', '-')) < 0:
                labels = labels[1:]
                locs = locs[1:]
        if len(set(results[x])) > 10:
            new_locs = []
            new_labels = []
            for i in range(len(locs)):
                if i%2 == 0:
                    new_locs.append(locs[i])
                    new_labels.append(labels[i])
            plt.xticks(new_locs, new_labels)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_xlabel(desc4paper(x), fontsize=font_size_labels)
        ax.set_ylabel(desc4paper(y_), fontsize=font_size_labels)
        fig.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        labels = [method2legend[x] if x in method2legend.keys() else x for x in labels]
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 0.5))
        if x == 'dataset':
            ticks, labels = plt.xticks()
            label_strings = [lab.get_text() for lab in labels]
            new_labels = [image_dataset2labels(s) for s in label_strings]
            plt.xticks(ticks, new_labels)
        if save_path is not None:
            plt.savefig(save_path + '/' + file_desc + y_ + '.pdf', bbox_inches="tight")
            plt.savefig(save_path + '/' + file_desc + y_ + '.png', bbox_inches="tight")
        ax.get_legend().remove()
        if save_path is not None:
            plt.savefig(save_path + '/' + file_desc + y_  + '_no_legend.pdf', bbox_inches="tight")
            plt.savefig(save_path + '/' + file_desc + y_  + '_no_legend.png', bbox_inches="tight")


def plot(results, level=0.02, save_path=None, x=None, y=None, hue='Type', epsilon=0.01, showmeans=True):
    global hue_order_g
    font_size = 16
    font_size_labels = 20
    font_size_legend = 20
    results = results[results['level'].astype(str) == str(level)]
    level = float(level)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    if y is None or len(y) == 0:
        y = ['Power', 'FDR']
    if x is None:
        # add dummy x
        x = 'dummy'
        results['dummy'] = 'ALL'
    for y_ in y:
        fig = plt.figure(figsize=([5, 4]))
        ax = fig.add_subplot(111)
        if x == 'dummy':
            # add spaces between the boxes
            sns.boxplot(results, x=x, hue=hue, y=y_, palette=palette4fig, ax=ax, hue_order=hue_order_g, width=0.5)
            ax.set_xlim(-0.6, 0.6)  # For 6 hue boxes
        else:
            sns.boxplot(results, x=x, hue=hue, y=y_, palette=palette4fig, ax=ax, hue_order=hue_order_g)
        if y_ == 'FDR':
            ax.axhline(level, color='black', linestyle='dashed', label=r'$\alpha$')
            ax.axhline(level + epsilon, color='gray', linestyle='dashed', label=r'$\alpha + \varepsilon$')
        if showmeans:
            hue_levels = hue_order_g
            if x is None:
                category_levels = [0]
            else:
                category_levels = list(results[x].unique())
            offsets = np.linspace(-0.2, 0.2, len(hue_levels))  # Adjust for dodge
            if x == 'alpha':
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
                        mean_value = subset[y_].mean()  # Compute mean
                        x_position = i + offsets[j]  # Adjust x position for dodge
                        ax.scatter(x_position, mean_value, color=palette4fig[hue_], edgecolor='black', s=100, zorder=3, marker=markers4fig[hue_])
    
        plt.rc('legend', fontsize=font_size_legend)
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_ylabel(y_, fontsize=font_size_labels)
        if x == 'dummy':
            ax.set_xticks([])
            ax.set_xlabel('')
        else:
            ax.set_xlabel(desc4paper(x), fontsize=font_size_labels)
        fig.tight_layout()
        handles, labels = ax.get_legend_handles_labels()
        labels = [method2legend[x_] if x_ in method2legend.keys() else x_ for x_ in labels]
        plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1.04, 0.5))
        plt.savefig(save_path + '/' + y_ + '.pdf', bbox_inches="tight")
        plt.savefig(save_path + '/' + y_ + '.png', bbox_inches="tight")
        ax.get_legend().remove()
        plt.savefig(save_path + '/' + y_ + '_no_legend.pdf', bbox_inches="tight")
        plt.savefig(save_path + '/' + y_ + '_no_legend.png', bbox_inches="tight")

