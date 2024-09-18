# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:12:05 2023

GAPy function parameters approximation

@author: klaat
"""

import numpy as np



from gapy import GAPy,selection_rank

#%% data creation

NOISE_PCT=10
NUM_POINTS=50

def mysterious_function(x,pars):
    # A sin (B(x-C)) + D
    collector=[]
    for xi in x:
        collector.append(pars['A']*np.sin(pars['B']*(xi+pars['C']))+pars['D'])
    return np.array(collector)
        
        
x_values=np.linspace(-10,10,num=NUM_POINTS)

actual_pars={'A':1,
             'B':.75,
             'C':2,
             'D':1}

y_pure=mysterious_function(x_values, actual_pars)
y_values=np.array([yi+np.random.randn()*NOISE_PCT/100*yi for yi in y_pure])



#%% Using GAPy to find suitable set of parameters

NUM_CROMOSOMES=100


# initial population
initial_population=[]
for i in range(NUM_CROMOSOMES):
    initial_population.append({'A':np.random.randint(0,5),
                               'B':np.random.randint(0,5),
                               'C':np.random.randint(0,5),
                               'D':np.random.randint(0,5)})

# FITNESS FUNCTION
def fit_fun_params_tuning(solution,xvalues,y_actual,base_function):
    y_pred=base_function(xvalues,solution)
    return np.mean(np.abs(y_pred-y_actual))



# CROSSOVER AND MUTATION
def mutation_params_tuning(solution):
    muted_param=np.random.choice(['A','B','C','D'])
    solution[muted_param]=solution[muted_param]+np.random.randn()*0.5
    return solution

def crossover_basic_params_tuning(s1,s2):
    res={}
    for param in ['A','B','C','D']:
        res[param]=(s1[param]+s2[param])/2
    return res

def crossover_weighted_params_tuning(s1,s2,fit1,fit2):
    w1=1-fit1/(fit1+fit2)
    w2=1-w1
    res={}
    for param in ['A','B','C','D']:
        res[param]=w1*s1[param]+w2*s2[param]
    return res



# OPTIMIZER

params_optimizer=GAPy(initial_population,
          selection_function=selection_rank,
          fitness_function=lambda x:fit_fun_params_tuning(x,x_values,y_values,mysterious_function),
          crossover_function=crossover_weighted_params_tuning,
          mutation_function=mutation_params_tuning)


soluzione=params_optimizer.optimize(max_generations=600,
                                    elite=3,
                                    max_time=10,
                                    patience=100,
                                    print_frequency=10)


y_pred=mysterious_function(x_values,soluzione.coded)


#%% Plot con BOKEH

from bokeh.plotting import figure
from bokeh.io import output_file, show


p = figure(title='Fitness trend',
           width=800,height=500,
           x_axis_label='Generations',
           y_axis_label='Fitness')
p.line(x=list(range(len(params_optimizer.status_summary['current_best_fitness']))),
       y=params_optimizer.status_summary['current_best_fitness'],
       line_width=3,
       legend_label='Best')
p.circle(x=list(range(len(params_optimizer.status_summary['mean_fitness']))),
       legend_label='Average',
       y=params_optimizer.status_summary['mean_fitness'],
       color='red',alpha=0.4, size=8)

output_file('gatrend.html')
show(p)



p = figure(title='Function fitting results',
           background_fill_color="#fafafa",           
           width=1000,height=600)
p.line(x=x_values,
       y=y_pure,
       line_width=4,
       legend_label='Theoretical')
p.scatter(x=x_values,
         y=y_values,
       legend_label='Actual (noisy)',      
       color='red',alpha=0.6,size=10,marker='triangle')
p.scatter(x=x_values,
         y=y_pred,
       legend_label='Estimated points',      
       color='green',alpha=0.6,size=10)
p.legend.click_policy="hide"
output_file('funfit_result.html')
show(p)
    

