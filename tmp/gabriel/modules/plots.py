import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_samples
import numbers
import matplotlib


def plot_clusters(data, attribute_names, plots_path, show_plots, wo=False):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18.0

    fig, ax = plt.subplots()

    for experiment in data.keys():
        experiment_type = (experiment.split('_')[0] if not(
            wo) else experiment.split('_')[0] + '-wo')
        for i, attr in enumerate(attribute_names[experiment_type]):
            plt.title('Experiment: %s - attribute: %s' % (experiment, attr))

            plot_data = []
            labels = []
            n = 0
            for cluster in data[experiment]['clusters']:
                cluster = np.array(cluster)
                plot_data.append(cluster[:, i])
                labels.append('Cluster %d' % (n + 1))
                n += 1

            plt.boxplot(plot_data, False, '', labels=labels)
            plot_file = plots_path + experiment + '_' + attr + '.png'
            plt.savefig(plot_file)
            print('Graph %s saved.' % plot_file)
            if show_plots:
                plt.show()
            plt.clf()


def plot_inertia(data, file_name, cluster_list, show_plots):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18.0

    fig, ax = plt.subplots()

    plot_data = []
    labels = []

    data_sorted = sorted(data.items(), key=lambda x: str(x[0]))

    for iteration, (experiment, value) in enumerate(data_sorted):
        labels.append(experiment)
        plot_data.append(value['inertia'])

    groups = np.arange(len(data.keys()))
    width = 0.35

    pallete = (sns.color_palette("husl", len(cluster_list))) * \
        int(len(plot_data) / len(cluster_list))
    plt.bar(groups, plot_data, width,
            tick_label=labels, color=pallete)
    plt.xticks(groups, labels, rotation=90)
    plt.title("Inertia for each experiment")
    plt.savefig(file_name)
    print('Graph %s saved.' % file_name)
    if show_plots:
        plt.show()
    plt.clf()


def plot_silhouette_score(data, file_name, cluster_list, show_plots):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18.0

    fig, ax = plt.subplots()

    plot_data = []
    labels = []

    data_sorted = sorted(data.items(), key=lambda x: str(x[0]))

    for iteration, (experiment, value) in enumerate(data_sorted):
        labels.append(experiment)
        plot_data.append(value['silhouette_score'])

    groups = np.arange(len(data.keys()))
    width = 0.35

    pallete = (sns.color_palette("husl", len(cluster_list))) * \
        int(len(plot_data) / len(cluster_list))
    plt.bar(groups, plot_data, width,
            tick_label=labels, color=pallete)
    plt.xticks(groups, labels, rotation=90)
    plt.title("Silhouette Score for each experiment")
    plt.savefig(file_name)
    print('Graph %s saved.' % file_name)
    if show_plots:
        plt.show()
    plt.clf()


def plot_counts(data, cluster_list, plots_path, show_plots):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 12

    titles = {}

    for experiment in data.keys():
        titles[experiment.split('_')[0]] = "Clusters with the experiment \"" + experiment.split('_')[0] + \
            "\" running the k-means for k = " + str(cluster_list)

    data_sorted = sorted(data.items(), key=lambda x: x[0])

    for iteration, (experiment, value) in enumerate(data_sorted):
        plot_data = []
        labels = []
        i = 1
        for c in value['clusters']:
            plot_data.append(len(c))
            labels.append('Cluster ' + str(i))
            i += 1

        tam = len(cluster_list)
        if iteration % tam == 0:
            fig, axes = plt.subplots(1, tam)
            axes[iteration % tam].pie(plot_data, autopct='%1.1f%%')
            axes[iteration % tam].axis('equal')
            axes[iteration % tam].set_title("k = " + str(len(labels)))
        elif iteration % tam >= 1 and iteration % tam < tam - 1:
            axes[iteration % tam].pie(plot_data, autopct='%1.1f%%')
            axes[iteration % tam].axis('equal')
            axes[iteration % tam].set_title("k = " + str(len(labels)))
        else:
            axes[iteration % tam].pie(plot_data, autopct='%1.1f%%')
            axes[iteration % tam].axis('equal')
            axes[iteration % tam].set_title("k = " + str(len(labels)))
            file_name = plots_path + experiment.split('_')[0] + '_pie.png'
            plt.suptitle(titles[experiment.split('_')[0]], fontsize=20)
            plt.savefig(file_name)
            print('Graph %s saved.' % file_name)
            if show_plots:
                plt.show()
            plt.clf()


def plot_distributions(data, attribute_names, plots_path, show_plots, norm):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18.0

    for attr in attribute_names:
        if attr == 'all' or attr == 'kda':
            continue
        if norm:
            ax = sns.distplot(data[attr])
        else:
            ax = sns.distplot(data[attr], kde=False)
        title = attr + ' distribution'
        if norm:
            title += ' - normalized'
        plt.suptitle(title, fontsize=20)
        file_name = plots_path + attr + '_dist'
        if norm:
            file_name += '_norm'
        file_name += '.png'
        plt.savefig(file_name)
        print('Graph %s saved.' % file_name)
        if show_plots:
            plt.show()
        plt.clf()


def plot_all_sep_distributions(data, attribute_names, plots_path, show_plots, norm):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18

    f, axarr = plt.subplots(3, 3, sharey=True)

    i = 0
    j = 0

    for attr in attribute_names:
        if norm:
            sns.distplot(data[attr], ax=axarr[i, j])
        else:
            sns.distplot(data[attr], ax=axarr[i, j], kde=False)
        title = attr + ' distribution'
        if norm:
            title += ' - normalized'
        axarr[i, j].set_title(title)
        j += 1
        if j == 3:
            j = 0
            i += 1
    if show_plots:
        plt.show()
    file_name = plots_path + 'all_sep_dist'
    if norm:
        file_name += '_norm'
    file_name += '.png'
    plt.savefig(file_name)
    print('Graph %s saved.' % file_name)
    plt.clf()


def plot_all_distributions(data, attribute_names, plots_path, show_plots, norm):
    # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18

    for attr in attribute_names:
        if norm:
            sns.distplot(data[attr], label=attr)
        else:
            sns.distplot(data[attr], label=attr, kde=False)

    title = 'All distributions'
    if norm:
        title += ' - normalized'
    plt.suptitle(title, fontsize=20)
    plt.legend()
    if show_plots:
        plt.show()
    file_name = plots_path + 'all_dist'
    if norm:
        file_name += '_norm'
    file_name += '.png'
    plt.savefig(file_name)
    print('Graph %s saved.' % file_name)
    plt.clf()


def plot_k_analysis(data, attribute_names, plots_path, show_plots):
     # config output images
    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 18.0

    for experiment in data:
        fig, ax = plt.subplots()
        plt.title("Inertia x K - %s" % experiment)
        plt.plot(data[experiment]['n_clusters'], data[experiment]['inertia'])
        file_name = plots_path + experiment + '_k_analysis.png'
        plt.savefig(file_name)
        print('Graph %s saved.' % file_name)
        if show_plots:
            plt.show()
        plt.clf()


def plot_f(kmeans_data, type_experiment, plots_path, show_plots):
    f = {}
    for experiment in kmeans_data:
        if experiment.split('_')[0] == 'all' or experiment.split('_')[0] == 'kda':
            f[experiment] = {}
            for k, cluster in enumerate(kmeans_data[experiment]['clusters']):
                f[experiment][k] = 0
                for player in cluster:
                    f[experiment][k] += (player[0] + player[2]) / player[1]
                f[experiment][k] /= len(cluster)

    plt.rcParams["figure.figsize"] = (25, 16)
    plt.rcParams['font.size'] = 12.0
    data = []
    labels = []
    for experiment in f:
        for k, cluster in enumerate(f[experiment]):
            data.append(f[experiment][k])
            labels.append(experiment + ' - C' + str(k + 1))

    pallete = sns.color_palette("husl", len(data))

    groups = np.arange(len(data))

    plt.bar(groups, data, 0.35, tick_label=labels, color=pallete)
    plt.xticks(groups, labels, rotation=90)
    plt.title("F for each cluster")
    file_name = plots_path + 'f-analysis-' + type_experiment + '.png'
    plt.savefig(file_name)
    plt.clf()
    print('Graph %s saved.' % file_name)


def plot_silhouette_analysis(data, attr_set, k, cluster_labels, silhouette_avg, file_name, show_plots, horizontal):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')

    sample_silhouette_values = silhouette_samples(
        data, cluster_labels, metric="euclidean")

    fig = plt.figure(figsize=(3.8, 2.3))
    plt.tight_layout()
    plt.rc('font', size=7)

    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = (sns.color_palette("husl", k))

        if horizontal:
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color[i], edgecolor=color[i], alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))
        else:
            plt.fill_between(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color[i], edgecolor=color[i], alpha=0.7)
            plt.text(y_lower + 0.5 * size_cluster_i, -0.05, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    if attr_set is not None:
        plt.title("The silhouette plot for the experiment " +
                  attr_set + "_" + str(k))

    if horizontal:
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.yticks([])
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        file_name += "_horizontal.pdf"
    else:
        plt.ylabel("The silhouette coefficient values")
        plt.xlabel("Cluster label")
        plt.axhline(y=silhouette_avg, color="red", linestyle="--")
        plt.yticks([-0.1, 0, 0.2, 0.4, 0.6])
        plt.xticks([])
        file_name += "_vertical.pdf"

    if show_plots:
        plt.show()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    print('Graph %s saved.\n' % file_name)


def radarplot(data, file_name, title=None, exclude=None, label=None, show_plots=False):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')

    candidates = list(data.columns)
    dimensions = []
    for i, c in enumerate(candidates):
        if isinstance(data.loc[0, c], numbers.Number):
            if exclude is not None:
                if c not in exclude:
                    dimensions.append(candidates[i])
            else:
                dimensions.append(candidates[i])

    dimensions = np.array(dimensions)
    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(5, 3.4))
    plt.rc('font', size=7)
    ax = fig.add_subplot(111, polar=True)
    colors_vec = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8",
                  "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#fabebe", "#008080"]
    pallete = sns.color_palette(colors_vec)
    for index, i in enumerate(data.index):
        values = data[dimensions].loc[i].values
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2,
                label=label[i], color=pallete[index])
        ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, dimensions)
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.legend(loc="center right", bbox_to_anchor=(1.32, 0.5),
               title="Centroid")  # bbox_to_anchor=(1.1, 0.5)
    if show_plots:
        plt.show()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    print('Graph %s saved.' % file_name)


def radarplot_top_k(data, file_name, exclude_list, title=None, show_plots=False, method=None, label=None):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.style.use('ggplot')

    if label is None:
        label = {}
        for index, player in enumerate(data.index):
            label[player] = 'Top ' + \
                str(index + 1) if method != 'avg' else 'Top ' + \
                str(player) + ' - avg'

    data = data.drop(exclude_list, axis=1)

    colunms_order = ['kills', 'hd', 'assists', 'hh',
                     'deaths', 'denies', 'lh', 'gpm', 'xpm']
    data = data.reindex(columns=colunms_order)

    dimensions = np.array(list(data.columns))
    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(5, 3.4))
    plt.rc('font', size=7)
    ax = fig.add_subplot(111, polar=True)
    colors_vec = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8",
                  "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#fabebe", "#008080"]
    pallete = sns.color_palette(colors_vec)
    for index, i in enumerate(data.index):
        values = data[dimensions].loc[i].values
        values = np.concatenate((values, [values[0]]))
        if label is None:
            ax.plot(angles, values, 'o-', linewidth=2,
                    label=label[i], color=pallete[index])
        else:
            ax.plot(angles, values, 'o-', linewidth=2,
                    label=label[index], color=pallete[index])
        ax.fill(angles, values, alpha=0.25)
    ax.set_ylim(top=1.0)
    ax.set_thetagrids(angles * 180/np.pi, dimensions)
    if title is not None:
        ax.set_title(title)
    ax.grid(True)
    plt.legend(loc='best', bbox_to_anchor=(1.1, 0.5))
    if show_plots:
        plt.show()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.01)
    plt.clf()
    print('Graph %s saved.' % file_name)
