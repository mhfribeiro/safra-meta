#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:41:12 2018

@author: marcos
"""
import argparse
from modules.data import read_data
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import sklearn.feature_selection as fs

def fitness(data, metric='euclidean', k=10, seed=None):
    if metric == 'variance':
        sel = fs.VarianceThreshold()
        sel.fit(data)
        return np.average(np.array(sel.variances_))
        
    if seed < 0:
        random_seed = None
    else:
        random_seed = seed
        
    km = KMeans(n_clusters=k, random_state=random_seed, n_jobs=-1)
    labels = km.fit_predict(data)
    if metric == 'euclidean':
        return silhouette_score(data, labels)
    elif metric == 'cosine':
        return silhouette_score(data, labels, metric='cosine')
    else:
        return km.inertia_
    
def solution_str(candidate):
    return '{' + ','.join(candidate['solution']) + '}: %f' % candidate['evaluation']
    
def save_solution(best, data, args):
    s = 'k;seed;metric;min_size;max_size;outliers\n'
    s += '%d;%d;%s;%d;%d;%d\n' % (args.k, args.seed, args.metric, args.mins, args.maxs, args.wo)
    
    s += 'best_solution\n'    
    
    head = ['evaluation']
    for i in range(len(data.columns)):
        head.append(data.columns[i])
    head = ';'.join(head)
    s += head + '\n'
    
    s += solution_str(best)
    
    if args.lang == 'pt':
        s = s.replace('.', ',')
    
    fp = open(args.csv_file, 'w')
    fp.write(s)
    fp.close()

def main():
    ### Parsing command line arguments
    parser = argparse.ArgumentParser(description='Exact algorithm for solving feature selection for the silhouette problem')
    parser.add_argument('csv_file', help='CSV file to save solution')
    parser.add_argument('--k', type=int, default=10, help='Number of clusters (default=10)')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed (default=-1). Use -1 for totally uncontrolled randomness')
    parser.add_argument('--lang', default='en', help='Whether use . or , as floating point number decimal separator in output. If lang=en, uses dot if lang=pt, uses comma (default=en)')
    parser.add_argument('--wo', action='store_true', help='Load data with outliers (default=False)')
    parser.add_argument('--mins', type=int, default=3, help='Minimum size of solution (default=3)')
    parser.add_argument('--maxs', type=int, default=6, help='Maximum size of solution (default=6)')
    parser.add_argument('--metric', choices=['euclidean', 'cosine', 'variance', 'inertia'], default='euclidean', help='Distance metric: euclidean | cosine | variance | inertia (default=euclidean)')
    args = parser.parse_args()
    
    ### Loading data
    if args.wo:
        data = read_data('df_data')
    else:
        data = read_data('df_data_pruned')
        
    ### Normalizing data
    for col in data.columns:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
    ### Generate all possible combinations
    cols = list(data.columns)
    combinations = []
    for r in range(args.mins, args.maxs+1):
        combs = itertools.combinations(cols, r)
        for c in combs:
            combinations.append(list(c))
            
    ### Find best solution
    count = 0
    if args.metric == 'inertia':
        best = {'evaluation': float('inf'), 'solution': []}
    else:
        best = {'evaluation': 0.0, 'solution': []}
    for c in combinations:
        candidate ={'evaluation': fitness(data[c], args.metric, args.k, args.seed), 'solution': c}
        print('\tTesting set =', solution_str(candidate), end=' ')
        if args.metric == 'inertia':
            improvement = candidate['evaluation'] < best['evaluation']
        else:
            improvement = candidate['evaluation'] > best['evaluation']
        if improvement:
            best = candidate
            print('new best!')
        else:
            print()
        count += 1
            
    print('\nBest solution found of %d tested:' % count)
    print('\tBest =', solution_str(best))
    
    save_solution(best, data, args)

if __name__=='__main__':
    main()
