# -*- coding: utf-8 -*-
"""

gapy/gapy.py
   _____            _____
  / ____|    /\    |  __ \
 | |  __    /  \   | |__) |_   _
 | | |_ |  / /\ \  |  ___/| | | |
 | |__| | / ____ \ | |    | |_| |
  \_____|/_/    \_\|_|     \__, |
                            __/ |
                           |___/


A minimalistic yet flexible
Genetic Algorithms implementation


@author: Marco Vannucci
         marco.vannucci AT santannapisa.it
"""

import numpy as np
import time
from datetime import timedelta
from inspect import signature
import copy
from typing import Any,Callable



#%% GAPY SOLUTION


class GAPy_solution:
    '''
    Class for the generic solution of the GA optimization problem
    '''
    
    def __init__(self,raw_solution:Any,initial_fitness:float=None):
        '''
        GAPy_solution constructor

        Args:
            raw_solution (Any): arbitrary raw-coding of a solution to the
            optimization problem.
            initial_fitness (float, optional): Fitness assigned to the
            solution. Defaults to None.

        Returns:
            None.

        '''
        self.fitness=initial_fitness
        self.coded=raw_solution
        self.changed=True
        
        
    def __str__(self):
        return str(self.coded)
    
    def _reset_fitness(self):
        '''
        Re-set the fitness value of the solution to None

        Returns:
            None.

        '''
        self.fitness=None


#%% GAPY GA operators


#   ================
#   SELECTION
#   ================

def selection_probabilistic(pop:list[GAPy_solution],num_selected:int)->np.ndarray[GAPy_solution]:
    '''
    Probabilistic selection of candidate solutions from a list of GAPy_solutions
    based on their fitness value. The lower the fitness, the higher the
    probability of selection.

    Args:
        pop (list[GAPy_solution]): list of solutions to pick from.
        num_selected (int): number of selected solutions.

    Returns:
        np.ndarray[GAPy_solution]: selected solutions.

    '''
    fitness_values=np.array([p.fitness for p in pop])
    probabilities=1-fitness_values/sum(fitness_values)  
    probabilities=probabilities/sum(probabilities)
    return np.random.choice(pop,num_selected,replace=False,p=probabilities)



def selection_tophalf(pop:list[GAPy_solution],num_selected:int)->np.ndarray[GAPy_solution]:
    '''
    Performs the selection of top-half ranked solutions from a list of GAPy_solutions
    based on their fitness value. Within top-half solutions, selection
    probability is uniform. The lower the fitness, the higher the probability
    of selection.

    Args:
        pop (list[GAPy_solution]): list of solutions to pick from.
        num_selected (int): number of selected solutions.

    Returns:
        np.ndarray[GAPy_solution]: selected solutions.

    '''
    # get fitness and sorted indices (the lower the better)
    fitness_values=np.array([p.fitness for p in pop])
    sorted_indices=np.argsort(fitness_values)
    # getting top-half
    selected_indices=sorted_indices[0:len(sorted_indices)//2]
    # select with uniform probability
    selectables=[pop[s] for s in selected_indices]
    return np.random.choice(selectables,num_selected,replace=False)

    
    


def selection_rank(pop:list[GAPy_solution],num_selected:int)->np.ndarray[GAPy_solution]:
    '''
    Performs the selection of solutions from a list of GAPy_solutions
    based on their ranking according to fitness value. The higher the ranking,
    the higher the probability of selection. Highest ranked solutions are those with LOWEST fitness

    Args:
        pop (list[GAPy_solution]): list of solutions to pick from.
        num_selected (int): number of selected solutions.

    Returns:
        np.ndarray[GAPy_solution]: selected solutions.

    '''
    fitness_values=np.array([p.fitness for p in pop])
    ranking=np.argsort(fitness_values)
    temp=np.zeros_like(ranking)
    
    for i,r in enumerate(ranking):
        temp[r]=max(ranking)-i+1
    
    probabilities=temp=temp/sum(temp)
    return np.random.choice(pop,num_selected,replace=False,p=probabilities)
    
    

#   ================
#   CROSSOVER
#   ================

def real_coded_average(p1:np.ndarray[float],p2:np.ndarray[float])->np.ndarray[float]:
    '''
    Average-type crossover for real-coded GA problems.
    Given 2 coded raw solutions returns their average.

    Args:
        p1 (np.ndarray[float]): parent 1.
        p2 (np.ndarray[float]): parent 2.

    Returns:
        np.ndarray[float]: offspring solution.

    '''
    return ((p1+p2)/2)



def real_coded_weight_average(p1:np.ndarray[float],p2:np.ndarray[float],fit1:float,fit2:float)->np.ndarray[float]:
    '''
    Weighted-Average-type crossover for real-coded GA problems.
    Given 2 coded raw solutions returns their weighted average according
    to their fitness. The lower the fitness, the higher the weight.

    Args:
        p1 (np.ndarray[float]): parent 1.
        p2 (np.ndarray[float]): parent 2.
        fit1 (float): fitness of parent 1.
        fit2 (float): fitness of parent 2.

    Returns:
        np.ndarray[float]: offspring solution.

    '''
    w1=1-fit1/(fit1+fit2)
    w2=1-w1
    return (w1*p1+w2*p2)




#   ================
#   MUTATION
#   ================


def real_coded_random_bounded_mutation(p:np.ndarray[float],bounds:np.ndarray[float],
                                       num_mutations:int=1)->np.ndarray[float]:
    '''
    Mutation within a range for real-coded GA.
    An arbitrary number of genes is mutated within arbitrary range of variability.

    Args:
        p (np.ndarray[float]): solotion to be mutated (raw coded).
        bounds (np.ndarray[float]): [R,2] array where R is the dimension of the
            solution. First column is the minimum value and second column 
            the maximum one for the corresponding gene
        num_mutations (int, optional): Number of genes of the solution
            to be mutated. Defaults to 1.

    Returns:
        p (np.ndarray[float]): mutated solution.

    '''
    #   MUTATION IS INPLACE
    #   now mutate its coded genes
    selected_genes_indices=np.random.choice(range(len(p)),size=num_mutations,replace=False)
    for g in selected_genes_indices:
        p[g]=np.random.rand()*(bounds[g,1]-bounds[g,0])+bounds[g,0]
    return p


def real_coded_local_bounded_mutation(p:np.ndarray[float],bounds:np.ndarray[float],
                                      max_mutation:np.ndarray[float],num_mutations=1)->np.ndarray[float]:
    '''
    Mutation within a range and a limitation for real-coded GA.
    An arbitrary number of genes is mutated within arbitrary range of variability
    and with a limitation in the variation of each gene.

    Args:
        p (np.ndarray[float]): solotion to be mutated (raw coded).
        bounds (np.ndarray[float]): [R,2] array where R is the dimension of the
            solution. First column is the minimum value and second column 
            the maximum one for the corresponding gene
        max_mutation (np.ndarray[float]): [R,] array where R is the dimension of the
            solution. Maximum change for the corresponding gene.
        num_mutations (int, optional): Number of genes of the solution
            to be mutated. Defaults to 1.

    Returns:
        p (np.ndarray[float]): mutated solution.

    '''
    #   MUTATION IS INPLACE
    #   now mutate its coded genes
    selected_genes_indices=np.random.choice(range(len(p)),size=num_mutations,replace=False)
    for g in selected_genes_indices:
        p[g]=p[g]+np.random.rand()*max_mutation[g]*np.random.choice([1,-1])
        if p[g]>bounds[g,1]:
            p[g]=bounds[g,1]
        if p[g]<bounds[g,0]:
            p[g]=bounds[g,0]            
    return p
    
        

#%% GAPY



class GAPy:
    '''
    
    Class for the GA engine.
    - Minimization of the fitness function value is assumed
    
    '''
    
    def __init__(self,initial_population:list[Any],
                 fitness_function:Callable[[Any],float],
                 selection_function:Callable[[list[GAPy_solution],int],list[GAPy_solution]],
                 crossover_function:Callable[[Any,Any],Any],
                 mutation_function:Callable[[Any,Any],Any]):
        '''
        Creates and initializes a GAPy object

        Args:
            initial_population (list[Any]): List of the chromosomes forming the
                initial population. Any type is supported.
            fitness_function (Callable[[Any],float]): fitness function. Being
                GAPy designed for minimization, the lower the fitness value
                for a solution, the better it is.
            selection_function (Callable[[list[GAPy_solution],int],list[GAPy_solution]]): GA selection function.
            crossover_function (Callable[[Any,Any],Any]): GA crossover function.
            mutation_function (Callable[[Any,Any],Any]): GA mutation function.

        Returns:
            None.

        '''
        
        self.pop_size=len(initial_population)
        
        self.status_summary={'generation':0,
                             'current_best_fitness':[],
                             'mean_fitness':[],
                             'global_best_fitness':[],
                             'best_solution_age':0}
                
        # creating the initial population
        self.population=[]
        for i in range(self.pop_size):
            self.population.append(GAPy_solution(initial_population[i]))
        #self.population=np.array(self.population)
            
        # linking functions
        # =================
        
        self.fitness_function=fitness_function
        self.selection_function=selection_function
        self.crossover_function=crossover_function
        self.mutation_function=mutation_function

        
                
        
    def describe_population(self):
        '''
        Shortly describes the GAPy actual population, including raw-coding
        of each solution and fitness value.

        Returns:
            None.

        '''
        for i,p in enumerate(self.population):
            print('[%d] %s '%(i+1,p.coded),end='')
            if p.fitness==None:
                print('F: na')
            else:
                print('F: %.3g'%p.fitness)
                      

    
    def _get_solutions_fitness(self)->list[float]:
        '''
        Returns the list of the fitness values of the solutions within
        the current population.

        Returns:
            list[float]: list of the fitness values of the solutions within
                the current population.

        '''
        evaluations=[sol.fitness for sol in self.population]
        return evaluations    
    
        
    def _eval_population(self):
        '''
        Calculates the fitness for all the samples in the current population.
        Fitness is stored in the .fitness field of each GAPy_solution object.

        Returns:
            None.

        '''
        for sol in self.population:
            if sol.changed:
                sol.fitness=self.fitness_function(sol.coded)
                sol.changed=False

            
            
    def optimize(self,max_generations:int=100,
                 target_fitness:float=0,
                 max_time:float=np.Inf,
                 patience:float=np.Inf,
                 elite:int=0,
                 recombination_rates:list[int,int,int]=[0.3, 0.7, 0.1],
                 print_frequency:int=1)->GAPy_solution:
        '''
        Starts the GA-based optimization and returns the obtained solution.

        Args:
            max_generations (int, optional): Maximum allowed generations for the GA. Defaults to 100.
            target_fitness (float, optional): Target fitness (minimization): if reached the algorithm stops. Defaults to 0.            
            max_time (float, optional): Maximum time for optimization, in seconds. Defaults to np.Inf.
            patience (float, optional): Maximum number of generations without improvement: if overcome the algorithm is stopped. Defaults to np.Inf.
            elite (int, optional): Number of elite candidate solutions. Defaults to 0.
            recombination_rates (list[int,int,int], optional): Recombination rates: rate of survivors as they are,
                rate of offspring solutions, rate of mutated elements. First 2 must sum up to 1. Last one <=1. Defaults to [0.3, 0.7, 0.1].
            print_frequency (int, optional): Frequency of search status reporting in generations . Defaults to 1.

        Returns:
            GAPy_solution: achieved solution at the termination of the optimization.

        '''
       
        
        optimization_start_time=time.time()
        
        # initializations
        generation=1
        
        # number of arguments of genetic operators
        nargs_crossover=len(signature(self.crossover_function).parameters)
        
        # first evaluation and associated operations
        self._eval_population()
        current_fitness=self._get_solutions_fitness()
        
        
        # for elite
        sorted_indices=np.argsort(current_fitness)    
        
        # best element and fitness
        best_fitness=np.min(current_fitness)
        #best_solution=self.population[sorted_indices[0]]
        
        
        # update dello status
        self.status_summary['generation']=generation
        self.status_summary['current_best_fitness'].append(np.min(current_fitness))
        self.status_summary['mean_fitness'].append(np.mean(current_fitness))
        self.status_summary['global_best_fitness'].append(best_fitness)
        

        print('[%d/%d] Ave: %6.3g  Curr.Best: %6.3g  Best: %6.3g [Age: %d]'%(generation,
                                                                max_generations,
                                                                np.mean(current_fitness),
                                                                np.min(current_fitness),
                                                                best_fitness,
                                                                self.status_summary['best_solution_age']
                                                                )) 
        
        
        
        
        #   =========================================
        #   MAIN CYCLE
        #   =========================================
        # in the cycle if terminal condition not satisfied
        
        best_fitness=np.Inf
        while generation<max_generations and \
            best_fitness>target_fitness and \
            self.status_summary['best_solution_age']<=patience and \
            time.time()-optimization_start_time<max_time:
            
            
            # EVOLVING THE POPULATION
            # =============================
            
            # determine recombination rates
            num_survivors=int(np.round(recombination_rates[0]*self.pop_size))-elite
            num_crossover=int(self.pop_size-(num_survivors+elite))
            num_mutations=int(np.round(recombination_rates[2]*self.pop_size))
                       
            
            # preparation of new population:
            # starts as an empty list of GAPy_solution
            new_population=[]
            
            
            # adding survivors
            # ====================
            survivors=self.selection_function(self.population, num_survivors)
            for s in survivors:
                s.changed=False
                new_population.append(copy.deepcopy(s))
            
            # adding offsprings
            # ====================
            for id_offspring in range(num_crossover):
                parents=self.selection_function(self.population,2)
                if nargs_crossover==2: # simple corssover function
                    new_candidate=self.crossover_function(parents[0].coded,parents[1].coded)
                else: # pass additional info
                    new_candidate=self.crossover_function(parents[0].coded,
                                                          parents[1].coded,
                                                          parents[0].fitness,
                                                          parents[1].fitness)
                    
                new_candidate=GAPy_solution(new_candidate)
                new_population.append(new_candidate)
                
            # preserve elite-solutions from mutation
            elite_solutions=[]
            for e in range(elite):
                elite_solutions.append(copy.deepcopy(self.population[sorted_indices[e]]))
            
            # perform mutations
            # ====================
            mutating_p=list(np.random.choice(new_population,num_mutations))
            for mp in mutating_p:
                mp.coded=self.mutation_function(mp.coded)
                mp.fitness=None
                mp.changed=True
                
            
            # after mutation copy the elite solutions
            for e in elite_solutions:
                new_population.append(e)

            
            # set new population
            self.population=new_population
            
            
            # NEW EVALUATION OF POPULATION
            # =============================
            generation+=1
            # evaluate population and related operations
            self._eval_population()
            current_fitness=self._get_solutions_fitness()
            sorted_indices=np.argsort(current_fitness)
            
            #self.describe_population()
            #input("Press Enter to continue...")
            
            if np.min(current_fitness)<best_fitness:
                # found a better solution
                self.status_summary['best_solution_age']=0
                best_fitness=np.min(current_fitness)
                best_solution=self.population[sorted_indices[0]].coded
            else:
                # maintain old best solution
                self.status_summary['best_solution_age']+=1
                
            # update dello status
            self.status_summary['generation']=generation
            self.status_summary['current_best_fitness'].append(np.min(current_fitness))
            self.status_summary['mean_fitness'].append(np.mean(current_fitness))
            self.status_summary['global_best_fitness'].append(best_fitness)
            
            if generation%print_frequency==0:
                print(f"[{generation:>6d}/{max_generations:<6d}] Ave.:{self.status_summary['mean_fitness'][-1]:6.3g} Curr.Best:{self.status_summary['current_best_fitness'][-1]:6.3g} Best:{self.status_summary['global_best_fitness'][-1]:6.3g} [AGE {self.status_summary['best_solution_age']}]")
                
                
        elapsed_time = timedelta(seconds=time.time()-optimization_start_time)
        print(f"Elapsed Time: {str(elapsed_time)}")
        return GAPy_solution(best_solution,best_fitness)
   
