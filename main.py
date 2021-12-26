import matplotlib.pyplot as plt
from numpy import mean
import numpy as np
import ga

no_variables = 2
pop_size = 80
crossover_rate = 40
mutation_rate = 40
no_generations = 200
lower_bounds = [-5.12, -5.12]
upper_bounds = [5.12, 5.12]
step_size = 0.2
rate = 15
pop = np.zeros((pop_size,no_variables))
for s in range(pop_size):
    for h in range(no_variables):
        pop[s,h] = np.random.uniform(lower_bounds[h],upper_bounds[h])

extended_pop = np.zeros((pop_size+crossover_rate+mutation_rate+2*no_variables*rate,pop.shape[1]))
#visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title('Red = Min      Blue = Average')
plt.xlabel("Iteration")
plt.ylabel("Objective function")
A = []
B = []
a=20 #adaptive restart
g=0
global_best = pop
k=0
while g <= no_generations:
    for i in range(no_generations):
        offspring1 = ga.crossover(pop, crossover_rate)
        offspring2 = ga.mutation(pop, mutation_rate)
        fitness = ga.objective_function(pop)
        offspring3 = ga.local_search(pop, fitness, lower_bounds, upper_bounds, step_size, rate)
        step_size = step_size*0.98
        if step_size < 0.01:
            step_size = 0.01
        extended_pop[0:pop_size] = pop
        extended_pop[pop_size:pop_size+crossover_rate] = offspring1
        extended_pop[pop_size+crossover_rate:pop_size+crossover_rate+mutation_rate] = offspring2
        extended_pop[pop_size+crossover_rate+mutation_rate:pop_size+crossover_rate+mutation_rate+2*no_variables*rate] = offspring3
        fitness = ga.objective_function(extended_pop)
        index = np.argmax(fitness)
        current_best = extended_pop[index]
        pop = ga.selection(extended_pop,fitness,pop_size)

        print("Generation: ", g, ", current best solution: ", current_best, ", current fitness value: ", 10e6-max(fitness))

        A.append(10e6-max(fitness))
        B.append(10e6-mean(fitness))
        g +=1

        if i >= a:
            if sum(abs(np.diff(A[g-a:g])))<=0.01:
                fitness = ga.objective_function(pop)
                index = np.argmax(fitness)
                current_best = pop[index]
                pop = np.zeros((pop_size, no_variables))
                for s in range(pop_size):
                    for h in range(no_variables):
                        pop[s, h] = np.random.uniform(lower_bounds[h], upper_bounds[h])
                pop[0] = current_best #keep the best
                step_size = 0.2
                global_best[k] = current_best
                k +=1
                break

        #Visualization
        ax.plot(A, color='r')
        ax.plot(B, color='b')
        fig.canvas.draw()
        ax.set_xlim(left=max(0, g - no_generations), right=g+3)
        if g > no_generations:
            break
    if g > no_generations:
        break

plt.show()
