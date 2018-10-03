# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 19:01:19 2018

@author: marcos
"""

from abc import ABC, abstractmethod
import copy
from operator import attrgetter
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

class Grasp(ABC):
    def __init__(self, items, alpha, max_iter, elite_size, max_no_improv=0.2, maximise=True, verbose=False):
        super(Grasp, self).__init__()
        self.maximise = maximise        
        
        self.items = items
        self.reset_rcl()
        
        self.alpha = alpha
        self.max_no_improv = int(max_no_improv * max_iter)
        self.max_iter = max_iter
        self.best = Solution()
        self.iteration = 0
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
    
    def construct_greedy_randomized(self):
        items = []
        self.reset_rcl()
        while len(self.rcl):
            self.update_rcl()
            i = random.randint(0, len(self.rcl)-1)
            item = self.rcl.pop(i)
            
            candidate = items + [item]
            if self.check_feasibility(candidate):
                items.append(item)
            else:
                self.rcl.append(item)
            
            self.reevaluate_rcl_items()
            
        solution = Solution(items=items, maximise=self.maximise)
        solution.evaluation = self.cost(solution)
        
        return solution
    
    def reset_rcl(self):
        self.rcl = copy.deepcopy(self.items)
        
    def update_rcl(self):
        """rcl_max = max(self.rcl, key=attrgetter('insertion_cost')).insertion_cost
        rcl_min = min(self.rcl, key=attrgetter('insertion_cost')).insertion_cost
        
        if self.maximise:
            threshold = rcl_max - self.alpha*(rcl_max - rcl_min)
            self.rcl = [item for item in self.rcl if item.insertion_cost >= threshold]
        else:
            threshold = rcl_min + self.alpha*(rcl_max - rcl_min)
            self.rcl = [item for item in self.rcl if item.insertion_cost <= threshold]"""
        n = min(self.alpha, len(self.rcl))
        self.rcl.sort(key=attrgetter('insertion_cost'), reverse=self.maximise)
        self.rcl = self.rcl[:n]
    
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
        
    def run(self):
        count_no_improv = 0
        while self.iteration < self.max_iter:
            if self.verbose:
                print('===============================================')
                print('GRASP Iteration %d:' % (self.iteration+1))
            candidate = self.construct_greedy_randomized()
            if self.verbose:
                print('\tSolution constructed. Evaluation = %f' % candidate.evaluation)
            self.check_elite(candidate)
            
            candidate = self.local_search(candidate)
            self.check_elite(candidate)
            
            if self.improvement(candidate, self.best):
                if self.verbose:
                    print('\n\t\tNew best! Evaluation: %f' % candidate.evaluation)
                self.best = candidate
                count_no_improv = 0
            else:
                count_no_improv += 1
                
            self.iteration += 1
            if self.verbose:
                print('\tBest=%f' % (self.best.evaluation))
            if count_no_improv == self.max_no_improv:
                if self.verbose:
                    print('=============================================')
                return False
                
        if self.verbose:
            print('=====================================================')
            
        return True
            
    def get_best(self):
        return self.best
        
    def get_iteration(self):
        return self.iteration
        
    def get_elite(self):
        return self.elite