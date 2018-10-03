#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:52:54 2018

@author: bruno
"""

from abc import ABC, abstractmethod
from operator import attrgetter
import copy
import numpy as np
import random

class Item(ABC):
    def __init__(self, item_id, insertion_cost):
        super(Item, self).__init__()
        self.id = item_id
        self.insertion_cost = insertion_cost
    
    @abstractmethod
    def __repr__(self):
        pass

class Solution(object):
    def __init__(self, items=[], evaluation=None, maximise=True):
        super(Solution, self).__init__()
        self.maximise = maximise
        self.items = items
        if evaluation is not None:
            self.evaluation = evaluation
        elif self.maximise:
            self.evaluation = -float('inf')
        else:
            self.evaluation = float('inf')
        self.compute_hash()
        
    def get_hash(self):
        self.compute_hash()
        return self.hash
    
    def compute_hash(self):
        if len(self.items):
            self.hash = copy.deepcopy(self.items[0].id)
            for i in range(1, len(self.items)):
                self.hash += self.items[i].id
            self.hash = ''.join([str(item) for item in self.hash])
        else:
            self.hash = ''
        
    def __repr__(self):
        return repr((self.evaluation, self.items))

class VNS(ABC):
    def __init__(self, items, min_size, max_size, n_items, n_neighborhood, max_iter, elite_size, const, max_no_improv=0.2, maximise=True, verbose=False):
        super(VNS, self).__init__()
        self.maximise = maximise        
        self.const = const
        
        self.items = items
        self.create_cl()
                
        self.min_size = min_size
        self.max_size = max_size
        self.n_items = n_items
        self.n_neighborhood = n_neighborhood
        self.max_no_improv = int(max_no_improv * max_iter)
        self.max_iter = max_iter
        self.best = Solution()
        self.iteration = 0
        self.k_neighborhood = 0
        self.best_iteration = 0
        self.ls_count = 0
        self.elite_size = elite_size
        self.elite = []
        
        self.verbose = verbose
        
    @abstractmethod
    def cost(self, solution):
        pass
    
    @abstractmethod
    def get_neighbor(self, solution):
        pass
    
    @abstractmethod
    def check_feasibility(self, solution):
        pass
    
    @abstractmethod
    def reevaluate_rcl_items(self):
        pass
    
    def construct_greedy(self):
        items = []
        n = random.randint(self.min_size, self.max_size) # quantidade de itens na solucao inicial
        items = self.build_solution(n)
        solution = Solution(items=items, maximise=self.maximise)
        solution.evaluation = self.cost(solution)        
        
        return solution
    
    def build_solution(self, n):
        self.update_cl()
        s = copy.deepcopy(self.cl[:n])
        return s    
    
    def update_cl(self):
        self.cl = copy.deepcopy(self.items)
        self.cl.sort(key=attrgetter('insertion_cost'), reverse=self.maximise)
        
    def create_cl(self):
        self.cl = copy.deepcopy(self.items)
        self.cl.sort(key=attrgetter('insertion_cost'), reverse=self.maximise)
    
    def improvement(self, candidate, reference):
        if self.maximise:
            return candidate.evaluation > reference.evaluation
        
        return candidate.evaluation < reference.evaluation
    
    def local_search(self, solution):
        self.ls_count = 0
        while self.ls_count < self.max_no_improv:
            if self.verbose:
                print('\tLocal Search. Attempt #%d' % (self.ls_count+1))
            candidate = Solution(items=self.get_neighbor(solution))
            candidate.evaluation = self.cost(candidate)
            if self.improvement(candidate, solution):
                self.ls_count = 0
                if self.verbose:
                    print('\t\tSearch reseted. Improved to %f' % candidate.evaluation)
                solution = candidate
            else:
                self.ls_count += 1
        return solution
        
    def check_elite(self, solution):
        hashes = [obj.get_hash() for obj in self.elite]
        if solution.get_hash() not in hashes:
            if len(self.elite) < self.elite_size:
                self.elite.append(copy.deepcopy(solution))
                self.elite.sort(key=attrgetter('evaluation'), reverse=self.maximise)
            else:
                if self.maximise:
                    lower_bound = min(self.elite, key=attrgetter('evaluation')).evaluation
                    if solution.evaluation > lower_bound:
                        self.elite.pop(len(self.elite)-1)
                        self.elite.append(copy.deepcopy(solution))
                        self.elite.sort(key=attrgetter('evaluation'), reverse=self.maximise)
                else:
                    lower_bound = max(self.elite, key=attrgetter('evaluation')).evaluation
                    if solution.evaluation < lower_bound:
                        self.elite.pop(len(self.elite)-1)
                        self.elite.append(copy.deepcopy(solution))
                        self.elite.sort(key=attrgetter('evaluation'), reverse=self.maximise)
      
    def mean_std_elite(self):
        values = []
        for x in self.elite:
            values.append(x.evaluation)
        return round(np.mean(values), 5), round(np.std(values), 5)
        
    def get_vector(self, solution):
        vector = copy.deepcopy(solution.items[0].id)
        for i in range(1, len(solution.items)):
            vector += solution.items[i].id
            
        return vector
    
    def create_ones_zeros(self, vector):
        zeros = []
        ones = []
        for i,item in enumerate(vector):
            if item:
                ones.append(i)
            else:
                zeros.append(i)
                
        return ones, zeros    

    def random_choose(self):
        r = random.uniform(0.0, 1.0)
        return r
    
    def random_k_neighbor(self): # Shaking, pick a random neighbor from the k-th neighborhood
        number = len(self.best.items)
        vector = self.get_vector(self.best)
        ones, zeros = self.create_ones_zeros(vector)        
        #print('vector inicial', vector)
        for i in range(0, self.k_neighborhood+1):
            flip_index = random.choice(zeros)
            vector[flip_index] = 1
            a = zeros.pop(zeros.index(flip_index))
            flip_index = random.choice(ones)
            vector[flip_index] = 0
            b = ones.pop(ones.index(flip_index))            
            ones.append(a)
            zeros.append(b)
            
            r = self.random_choose()

            if (r >= 0.50 and r < 0.75) and number < self.max_size: # add item
                flip_index = random.choice(zeros)
                vector[flip_index] = 1
                a = zeros.pop(zeros.index(flip_index))
                ones.append(a)
                number += 1
            elif (r >= 0.75) and number > self.min_size: # del item
                flip_index = random.choice(ones)
                vector[flip_index] = 0
                a = ones.pop(ones.index(flip_index))
                zeros.append(a)
                number -= 1
                
        #print('vector final, ', vector)
        new_candidate = Solution(items=self.items_from_vector(vector))
        new_candidate.evaluation = self.cost(new_candidate)
        
        return new_candidate
        
    def run(self):        
        count_no_improv = 0      
        self.best = self.construct_greedy()
        if self.verbose:
            print('\tSolution constructed: ', self.best)
        self.check_elite(self.best)
                
        while self.iteration < self.max_iter:
            self.k_neighborhood = 0
            while self.k_neighborhood < self.n_neighborhood:
                if self.verbose:
                    print('===============================================')
                    print('VNS k_neighborhood %d:' % (self.k_neighborhood+1))
                # candidate = shaking best
                candidate = self.random_k_neighbor()
                self.check_elite(candidate)            
                candidate = self.local_search(candidate)
                self.check_elite(candidate)
                
                if self.improvement(candidate, self.best):
                    if self.verbose:
                        print('\n\t\tNew best! Evaluation: %f' % candidate.evaluation)
                    #self.best_iteration = self.iteration 
                    self.best = candidate
                    self.k_neighborhood = 0
                    count_no_improv = 0      
                else: # Next neighbor
                    count_no_improv += 1      
                    self.k_neighborhood += 1
                    
                if self.verbose:
                    print('\tBest=%f' % (self.best.evaluation))
                if self.k_neighborhood >= 3 or count_no_improv == self.max_no_improv: # Passed neighbor
                    if self.verbose:
                        print('=============================================')
                    return False
                self.iteration += 1    
            
        if self.verbose:
            print('=====================================================')
            
        return True
            
    def get_best(self):
        return self.best
        
    def get_iteration(self):
        return self.iteration + 1
    
    def get_best_iteration(self):
        return self.best_iteration + 1
        
    def get_elite(self):
        return self.elite