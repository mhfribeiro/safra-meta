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
from operator import attrgetter
import time

### Function for formatting time in human readable format
def get_formatted_time(s):
    decimal_part = s - int(s)
    
    s = int(s)
    
    seconds = s % 60

    s = s // 60
    minutes = s % 60

    s = s // 60
    hours = s % 24

    days = s // 24

    if days:
        d = '%dd ' % days
    else:
        d = ''

    return '%s%02dh%02dm%02d.%ds' % (d, hours, minutes, seconds, decimal_part)

class Problem(object):
    def __init__(self, data, corr_threshold, min_size, max_size, metric='cosine', elite_size=10):
        self.data = data
        self.corr_threshold = corr_threshold
        self.min_size = min_size
        self.max_size = max_size
        self.metric = metric
        self.attributes = list(self.data.columns)
        self.restrictions = []
        self.elite_size = elite_size
        self.elite = []
        
        self.create_restrictions()

    def create_restrictions(self):
        corr = self.data.corr()
        for i in range(0, len(self.attributes)-1):
            for j in range(i+1, len(self.attributes)):
                if abs(corr[self.attributes[i]][self.attributes[j]]) >= self.corr_threshold:
                    restriction = np.zeros(len(self.attributes), dtype=int)
                    restriction[i] = 1
                    restriction[j] = 1
                    self.restrictions.append(restriction)
        self.restrictions = np.array(self.restrictions)
    
    def count_violations(self, solution):
        n_selected = np.sum(solution)
        if n_selected < self.min_size or n_selected > self.max_size:
            return len(self.restrictions)+1
        else:
            violations = 0
            for i in range(len(self.restrictions)):
                if np.sum(np.bitwise_and(solution, self.restrictions[i])) >= 2:
                    violations += 1
                    
        return violations
    
    def manage_elite(self, individual):
        print('\tTesting set =', individual, end=' ')
        if len(self.elite) < self.elite_size:
            self.elite.append(individual)
            if self.metric == 'inertia':
                self.elite.sort(key=attrgetter('evaluation'), reverse=False)
            else:
                self.elite.sort(key=attrgetter('evaluation'), reverse=True)
            print('Elite updated!')
        elif self.metric == 'inertia':
            lower_bound = max(self.elite, key=attrgetter('evaluation')).evaluation
            if individual.evaluation < lower_bound:
                self.elite.pop(len(self.elite)-1)
                self.elite.append(individual)
                self.elite.sort(key=attrgetter('evaluation'), reverse=False)
                print('Elite updated!')
            else:
                print()
        else:
            lower_bound = min(self.elite, key=attrgetter('evaluation')).evaluation
            if individual.evaluation > lower_bound:
                self.elite.pop(len(self.elite)-1)
                self.elite.append(individual)
                self.elite.sort(key=attrgetter('evaluation'), reverse=True)
                print('Elite updated!')
            else:
                print()

class Individual(object):
    def __init__(self, attributes, problem, k=10, seed=None):
        self.problem = problem
        self.data = self.problem.data
        self.attributes = attributes
        self.k = k
        self.metric = self.problem.metric
        if seed < 0:
            self.random_seed = None
        else:
            self.random_seed = seed
        
        attribute_map = list(self.data.columns)
        self.solution = np.zeros(len(attribute_map), dtype=int)
        for i, item in enumerate(attribute_map):
            if item in self.attributes:
                self.solution[i] = 1
        
        self.violations = self.problem.count_violations(self.solution)
        self.feasible = self.violations == 0
                
    def __repr__(self):
        if self.feasible:
            feasibility = 'Feasible'
        else:
            feasibility = 'Unfeasible'
        return '{' + ','.join(self.attributes) + '}: %f (%s: %d violations)' % (self.evaluation, feasibility, self.violations)
    
    def get_csv(self):
        return ','.join(self.attributes) + ';%f;%d;%d;%d\n' % (self.evaluation, self.feasible,
                       len(self.attributes), self.violations)
        
    def getOrder(self, x):
        if x == 0.0:
            return 0.0
        return 10**np.floor(np.log10(np.abs(x)))
    
    def fitness(self):
        data_projected = self.data[self.attributes]
        
        if self.metric == 'variance':
            sel = fs.VarianceThreshold()
            sel.fit(data_projected)
            return np.average(np.array(sel.variances_)) - np.log(1 + self.violations)
        
        km = KMeans(n_clusters=self.k, random_state=self.random_seed, n_jobs=-1)
        labels = km.fit_predict(data_projected)
        if self.metric == 'euclidean':
            return silhouette_score(data_projected, labels) - np.log(1 + self.violations)
        elif self.metric == 'cosine':
            return silhouette_score(data_projected, labels, metric=self.metric) - np.log(1 + self.violations)
        else:
            inertia = km.inertia_
            order = self.getOrder(inertia)
            return inertia + self.violations * order*10**2
        
    def evaluate(self):
        self.evaluation = self.fitness()
        self.problem.manage_elite(self)

class Logger(object):
    def __init__(self, args, maximise, elapsed_time):
        self.head = 'k;seed;metric;min_size;max_size;outliers;threshold;elite_size;elapsed_time\n'
        self.head += '%d;%d;%s;%d;%d;%d;%f;%d;%f\n' % (args.k, args.seed,
                    args.metric, args.mins, args.maxs, args.wo, args.threshold,
                    args.elite_size, elapsed_time)
        self.elite_size = args.elite_size
        self.lang = args.lang
        self.file = args.csv_file
        self.maximise = maximise

    def write_content(self, solution_set):
        self.body += 'Solution;Evaluation;Feasible;Num. of Attr.;Restrictions Violated\n'
    
        for individual in solution_set:
            self.body += individual.get_csv()
    
        if self.lang == 'pt':
            self.body = self.body.replace('.', ',')
            
        self.write()

    def save_elite(self, elite):
        self.body = 'Top %d solutions\n' % self.elite_size
        self.write_content(elite)
        
    def save_log(self, solutions):
        solutions.sort(key=attrgetter('evaluation'), reverse=self.maximise)
        file_tmp = self.file
        self.file = self.file[:self.file.rfind('.')] + '_log' + self.file[self.file.rfind('.'):]

        self.body = 'All solutions found\n'
        self.write_content(solutions)
        
        self.file = file_tmp
    
    def write(self):
        content = self.head + self.body
        fp = open(self.file, 'w')
        fp.write(content)
        fp.close()

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(
        description='Exact algorithm for solving feature selection for the silhouette problem')
    parser.add_argument('csv_file', help='CSV file to save solution')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of clusters (default=10)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (default=-1). Use -1 for totally uncontrolled randomness')
    parser.add_argument('--lang', default='en',
                        help='Whether use . or , as floating point number decimal separator in output. If lang=en, uses dot if lang=pt, uses comma (default=en)')
    parser.add_argument('--wo', action='store_true',
                        help='Load data with outliers (default=False)')
    parser.add_argument('--mins', type=int, default=3,
                        help='Minimum size of solution (default=3)')
    parser.add_argument('--maxs', type=int, default=6,
                        help='Maximum size of solution (default=6)')
    parser.add_argument('--metric', choices=['euclidean', 'cosine', 'variance', 'inertia'], default='euclidean',
                        help='Distance metric: euclidean | cosine | variance | inertia (default=euclidean)')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Threshold value (defaut=0.75)')
    parser.add_argument('--elite_size', type=int, default=10,
                        help='Top n best solutions to be saved (defaut=10)')
    parser.add_argument('--save_log', '-l', action='store_true',
                        help='Save log with all solutions. The file name is the same from csv_file argument, with _log before extension (default=False)')
    args = parser.parse_args()

    # Loading data
    if args.wo:
        data = read_data('df_data')
    else:
        data = read_data('df_data_pruned')

    # Normalizing data
    for col in data.columns:
        data[col] = (data[col] - data[col].min()) / \
            (data[col].max() - data[col].min())
            
    # Create problem
    problem = Problem(data, args.threshold, args.mins, args.maxs, args.metric, args.elite_size)

    # Generate all possible combinations
    cols = list(data.columns)
    all_solutions = []
    initial_time = time.time()
    for r in range(2, args.maxs+1):
        combs = itertools.combinations(cols, r)
        for c in combs:
            individual = Individual(list(c), problem, args.k, args.seed)
            individual.evaluate()
            all_solutions.append(individual)
    elapsed_time = time.time() - initial_time

    print('\nTop 10 best solution found of %d tested:' % len(all_solutions))

    for solution in problem.elite:
        print(solution)
        
    print('\nTotal elapsed time: %s' % (get_formatted_time(elapsed_time)))

    if args.metric == 'inertia':
        maximise = False
    else:
        maximise = True
    logger = Logger(args, maximise, elapsed_time)
    logger.save_elite(problem.elite)
    if args.save_log:
        logger.save_log(all_solutions)


if __name__ == '__main__':
    main()
