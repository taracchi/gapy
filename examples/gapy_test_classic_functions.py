# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:13:52 2023

@author: klaat
"""

from gapy import GAPy, real_coded_average, real_coded_random_bounded_mutation,selection_rank, \
    real_coded_weight_average, real_coded_local_bounded_mutation
import numpy as np

from bokeh.plotting import figure
from bokeh.io import output_file, show

#%% Test functions

DOMAIN_DIMENSION=20
DIMENSION_BOUNDS=[-5.12,5.12]

def sphere(x):
    return np.sum([xi**2 for xi in x])


#%% Genetic operators and parameters

dim_popolazione=30 


# initial population
prima_pop=np.random.uniform(low=DIMENSION_BOUNDS[0],high=DIMENSION_BOUNDS[1],
                            size=(dim_popolazione,DOMAIN_DIMENSION))


bounds=np.array([DIMENSION_BOUNDS]*dim_popolazione)
maxmut=(bounds[:,1]-bounds[:,0])/5

nedo=GAPy(prima_pop,
          selection_function=selection_rank,
          fitness_function=sphere,
          crossover_function=real_coded_average,
          mutation_function=lambda p:real_coded_local_bounded_mutation(p, bounds,maxmut,num_mutations=2))


soluzione=nedo.optimize(max_generations=5000,
                        elite=1,
                        target_fitness=0.001,
                        print_frequency=50,
                        recombination_rates=[0.3,0.7,0.2])



p = figure(title='Fitness trend',
           background_fill_color="#fafafa",
           width=800,height=600,
           x_axis_label='Generations',
           y_axis_label='Fitness')
p.line(x=list(range(len(nedo.status_summary['current_best_fitness']))),
       y=nedo.status_summary['current_best_fitness'],
       line_width=3,
       legend_label='Best')
p.line(x=list(range(len(nedo.status_summary['mean_fitness']))),
       legend_label='Average',
       y=nedo.status_summary['mean_fitness'],
       color='red',line_width=2)
#p.legend.click_policy="hide"
output_file('sample1.html')
show(p)
   


