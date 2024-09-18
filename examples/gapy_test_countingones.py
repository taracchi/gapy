# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:13:52 2023

@author: klaat
"""

from gapy import GAPy,GAPy_solution, real_coded_average, real_coded_weight_average,real_coded_random_bounded_mutation,selection_rank, selection_probabilistic
import numpy as np



#   Some fitness functions for the tests
def sample_fitness(sol):
    return sum(sol)


# Mutation
def binary_mutation(sol,num_mutations=2):
    selected_genes_indices=np.random.choice(range(len(sol)),size=num_mutations,replace=False)
    for g in selected_genes_indices:
        if sol[g]==0:
            sol[g]=1
        else:
            sol[g]=0
    return sol

# Crossover
def binary_crossover(sol1,sol2):
    sol_child=np.zeros_like(sol1)
    for i in range(len(sol1)):
        sol_child[i]=np.random.choice([sol1[i],sol2[i]])
    return sol_child


dimensione_dominio=300
dim_popolazione=100 


# initial population
prima_pop=np.round(np.random.rand(dim_popolazione,dimensione_dominio))


nedo=GAPy(prima_pop,
          selection_function=selection_rank,
          fitness_function=sample_fitness,
          crossover_function=binary_crossover,
          mutation_function=lambda x:binary_mutation(x,num_mutations=3))


soluzione=nedo.optimize(max_generations=300,
                        elite=1,
                        print_frequency=10,
                        recombination_rates=[0.2,0.8,0.15])




from bokeh.plotting import figure
from bokeh.io import output_file, show


p = figure(title='Fitness trend',
           width=800,height=500,
           x_axis_label='Generations',
           y_axis_label='Fitness')
p.line(x=list(range(len(nedo.status_summary['current_best_fitness']))),
       y=nedo.status_summary['current_best_fitness'],
       line_width=3,
       legend_label='Best')
p.circle(x=list(range(len(nedo.status_summary['mean_fitness']))),
       legend_label='Average',
       y=nedo.status_summary['mean_fitness'],
       color='red',alpha=0.4)

output_file('sample1.html')
show(p)
    
#%% Istogramma

from dataware.bokeh_utils import draw_histogram

h = figure(title='Last generation fitness distribution',
           width=800,height=500,
           x_axis_label='Fitness',
           y_axis_label='Count')


draw_histogram(h,nedo.status_summary['mean_fitness'],num_bins=10,alpha=0.8)


output_file('sample2.html')
show(h)


