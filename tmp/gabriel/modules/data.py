import numpy as np
from pprint import PrettyPrinter
import pandas as pd
import json
import os
import scipy.stats as sci


def normalizes(x):
    x_norm = []
    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)
    # Arrumar esse maximum - minimum para fazer só quando esses valores são escalares
    # if maximum - minimum == 0:
    #    for i in x:
    #        x_norm.append((i - minimum))
    # else:
    for i in x:
        x_norm.append((i - minimum) / (maximum - minimum))

    return x_norm, minimum, maximum


def de_normalize(x, minimum, maximum):
    x_de_norm = []

    for i in x:
        x_de_norm.append(list(i * (maximum - minimum) + minimum))

    return x_de_norm


def remove_outliers(data, c=2.698):
    new_data = {}
    new_data['all'] = []
    new_data['kda'] = []
    new_data['kills'] = []
    new_data['deaths'] = []
    new_data['assists'] = []
    new_data['denies'] = []
    new_data['gpm'] = []
    new_data['hd'] = []
    new_data['hh'] = []
    new_data['lh'] = []
    new_data['xpm'] = []

    outliers = []
    outliers_attr = []
    attributes = ["kills", "deaths",
                  "assists", "denies", "gpm", "hd", "hh", "lh", "xpm"]
    index_attr = {"kills": 0, "deaths": 1,
                  "assists": 2, "denies": 3, "gpm": 4, "hd": 5, "hh": 6, "lh": 7, "xpm": 8}

    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)
    iqr = sci.iqr(data, axis=0)

    for d in data:
        att_v = []
        att_v.append(d >= q1 - 1.5 * iqr)
        att_v.append(d <= q3 + 1.5 * iqr)
        validation_outiers = np.all(att_v, axis=0)

        if validation_outiers.all():
            for key in new_data.keys():
                if key == 'all':
                    new_data[key].append(d)
                elif key == 'kda':
                    new_data[key].append(d[0:3])
                else:
                    new_data[key].append([d[index_attr[key]]])
        else:
            outliers.append(d)
            attr = []

            for index, value in enumerate(validation_outiers):
                if not(value):
                    attr.append(attributes[index])
            outliers_attr.append(attr)

    print("\n================ Summary about outliers ================\n")
    print("Number of outliers = ", len(outliers_attr))
    teste = [0 for i in range(10)]
    for i in outliers_attr:
        teste[len(i)] += 1

    print("Number of outliers in 1, ..., 9 attributes: ")
    print(teste[1:])
    print()
    print("Number of outliers per attribute: ")
    count_out_att = {}
    count_out_att['kills'] = 0
    count_out_att['deaths'] = 0
    count_out_att['assists'] = 0
    count_out_att['denies'] = 0
    count_out_att['gpm'] = 0
    count_out_att['hd'] = 0
    count_out_att['hh'] = 0
    count_out_att['lh'] = 0
    count_out_att['xpm'] = 0

    for i in outliers_attr:
        for j in i:
            count_out_att[j] += 1

    for i in count_out_att.keys():
        print(i, count_out_att[i], sep=': ')
    print()
    return new_data, outliers, outliers_attr


def read_data(method='data'):
    if method == 'data':
        with open('files/data/data.json') as fh:
            data = json.load(fh)
        return data
    elif method == 'corr':
        with open('files/data/data_corr.json') as fh:
            data = json.load(fh)
        return data
    elif method == 'corr_pruned':
        with open('files/data/data_corr_pruned.json') as fh:
            data = json.load(fh)
        return data
    elif method == 'pruned':
        with open('files/data/data_pruned.json') as fh:
            data = json.load(fh)
        return data
    elif method == 'outliers':
        with open('files/data/outliers.json') as fh:
            outliers = json.load(fh)
        with open('files/data/outliers_attr.json') as fh:
            outliers_attr = json.load(fh)
        return outliers, outliers_attr
    elif method == 'k-means_experiments':
        with open('files/output_k-means_experiments/data/output_kmeans.json') as fh:
            kmeans = json.load(fh)
        with open('files/output_k-means_experiments/data_corr/output_kmeans_corr.json') as fh:
            kmeans_corr = json.load(fh)
        with open('files/output_k-means_experiments/data_pruned/output_kmeans_pruned.json') as fh:
            kmeans_pruned = json.load(fh)
        return kmeans, kmeans_corr, kmeans_pruned
    elif method == 'df_data':
        df = pd.read_json('files/data/data_df.json')
        return df
    elif method == 'df_data_pruned':
        df = pd.read_json('files/data/data_pruned_df.json')
        return df
    elif method == 'df_w_metrics_all':
        df = pd.read_csv('files/data/df_w_metrics_all.csv')
        return df
    elif method == 'df_w_metrics_kda':
        df = pd.read_csv('files/data/df_w_metrics_kda.csv')
        return df
    elif method == 'df_w_metrics_adg':
        df = pd.read_csv('files/data/df_w_metrics_adg.csv')
        return df
    elif method == 'df_w_metrics_g':
        df = pd.read_csv('files/data/df_w_metrics_g.csv')
        return df
    elif method == 'df_w_metrics_x':
        df = pd.read_csv('files/data/df_w_metrics_x.csv')
        return df


def create_data(input_file, corr=True, verbose=False):
    print('\nCreating input data from file %s...' % input_file)

    players = []
    data = {}
    data_corr = {}
    data['all'] = []
    data['kda'] = []
    data['kills'] = []
    data['deaths'] = []
    data['assists'] = []
    data['denies'] = []
    data['gpm'] = []
    data['hd'] = []
    data['hh'] = []
    data['lh'] = []
    data['xpm'] = []
    if corr:
        data_corr['kills-corr'] = []
        data_corr['deaths-corr'] = []
        data_corr['assists-corr'] = []
        data_corr['denies-corr'] = []
        data_corr['gpm-corr'] = []
        data_corr['hd-corr'] = []
        data_corr['hh-corr'] = []
        data_corr['lh-corr'] = []
        data_corr['xpm-corr'] = []

    fp = open(input_file, 'r')

    for l in fp:
        parts = l.strip().split()
        for i, p in enumerate(parts):
            parts[i] = int(p)
        if parts[4] >= 5:
            data['kda'].append(list(np.array(parts[1:4]) / parts[4]))
            data['all'].append(
                list(np.array(parts[1:4] + parts[5:]) / parts[4]))
            data['kills'].append(
                list(np.array([parts[1]]) / parts[4]))
            data['deaths'].append(list(np.array([parts[2]]) / parts[4]))
            data['assists'].append(list(np.array([parts[3]]) / parts[4]))
            data['denies'].append(list(np.array([parts[5]]) / parts[4]))
            data['gpm'].append(list(np.array([parts[6]]) / parts[4]))
            data['hd'].append(list(np.array([parts[7]]) / parts[4]))
            data['hh'].append(list(np.array([parts[8]]) / parts[4]))
            data['lh'].append(list(np.array([parts[9]]) / parts[4]))
            data['xpm'].append(list(np.array([parts[10]]) / parts[4]))
            # K, D, A, Npartidas, denies, gpm, hero_damage, hero_healing, LH, xp_p_min

    fp.close()

    data_json = json.dumps(data)
    f = open('files/data/data.json', 'w')
    f.writelines(data_json)
    f.close()

    data_pruned, outliers, outliers_attr = remove_outliers(data['all'])

    print('Saving data in JSON files...', end='')
    data_json = json.dumps(data_pruned)
    f = open('files/data/data_pruned.json', 'w')
    f.writelines(data_json)
    f.close()

    data_json = json.dumps(outliers)
    f = open('files/data/outliers.json', 'w')
    f.writelines(data_json)
    f.close()

    data_json = json.dumps(outliers_attr)
    f = open('files/data/outliers_attr.json', 'w')
    f.writelines(data_json)
    f.close()
    print('done.')
    print('Creating dataframes and saving in JSON files...', end='')
    # Create a dataframes with data and data pruned
    new_data = {}
    for key in data:
        if key != 'all' and key != 'kda':
            new_data[key] = []
            for element in data[key]:
                new_data[key].append(element[0])

    df = pd.DataFrame(new_data)
    df.to_json('files/data/data_df.json')

    new_data.clear()
    for key in data_pruned:
        if key != 'all' and key != 'kda':
            new_data[key] = []
            for element in data_pruned[key]:
                new_data[key].append(element[0])
    df = pd.DataFrame(new_data)
    df.to_json('files/data/data_pruned_df.json')
    print('done.')

    # Dynamically maps n least correlated attributes to each attribute
    correlation_map = []
    correlation_map_names = {}

    if corr:
        print('Creating corr data...', end='')
        # Mapping the database
        attr_positions = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10])
        matches_position = 4
        attr_names = np.array(['kills', 'deaths', 'assists',
                               'denies', 'gpm', 'hd', 'hh', 'lh', 'xpm'], dtype=str)

        # n least correlated atrributes and minimum matches per player
        n = 4
        min_matches = 5

        # normalizes database and discard maximum and minimum information (underline redirects to "nothing")
        dt, _, _ = normalizes(data['all'])

        # Computes correlation matrix (absolute values)
        corr_matrix = pd.DataFrame(dt).corr().abs().as_matrix()

        # Line count
        i = 0
        for attr_line in corr_matrix:
            # Sort indexes of matrix line by its values and get the indexes related to the n smallest values
            sorted_indexes = attr_line.argsort()[:n]

            correlation_map.append(attr_positions[sorted_indexes])
            correlation_map_names[attr_names[i]] = list(
                attr_names[sorted_indexes])

            i += 1

        fp = open(input_file, 'r')
        for l in fp:
            parts = l.strip().split()
            for i, p in enumerate(parts):
                parts[i] = int(p)
            if parts[matches_position] >= min_matches:
                for i, position in enumerate(attr_positions):
                    line = [parts[position]]
                    # print('Data for %s:' % (attr_names[i] + '-corr'), end=' ')
                    for other in correlation_map[i]:
                        line.append(parts[other])
                        l = list(attr_positions)
                        # print('%s (%d)' % (attr_names[l.index(other)], other), end=' ')
                    # print()
                    data_corr[attr_names[i] + '-corr'].append(
                        list(np.array(line) / parts[matches_position]))

                # print()
        fp.close()

        # Data corr pruned
        data_corr_pruned = {}
        for index, attribute in enumerate(correlation_map):
            data_corr_pruned[attr_names[index] + '-corr'] = []
            for data in data_pruned['all']:
                attr_line = []
                attr_line.append(data[index])
                for i in attribute:
                    attr_line.append(data[i - 1 if i < 4 else i - 2])
                data_corr_pruned[attr_names[index] + '-corr'].append(attr_line)

        data_json = json.dumps(data_corr)
        f = open('files/data/data_corr.json', 'w')
        f.writelines(data_json)
        f.close()

        data_json = json.dumps(data_corr_pruned)
        f = open('files/data/data_corr_pruned.json', 'w')
        f.writelines(data_json)
        f.close()

        print('done.')

        if verbose:
            pp = PrettyPrinter()
            pp.pprint(correlation_map_names)
            print()

    print('Finished.\n')
    return correlation_map, correlation_map_names
