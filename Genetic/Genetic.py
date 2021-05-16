import numpy as np
import matplotlib.pyplot as plt
import gym

env=gym.make("FrozenLake-v0")

population_size=50
gene_size=env.observation_space.n
gene_options=4
generation_number=20
elite_pop_size=4
n_episode=100
mutation_chance=0.05

def cross_over_with_pieces_of_parents(parent1,parent2):
    pos=np.random.randint(gene_size)
    child1=np.concatenate((parent1[:pos], parent2[pos:]),axis=0)
    child2=np.concatenate((parent1[pos:], parent2[:pos]),axis=0)
    for i in np.arange(gene_size):
        if (np.random.uniform() < mutation_chance):
            child1[i] = np.random.randint(4)
            
    for i in np.arange(gene_size):
        if (np.random.uniform() < mutation_chance):
            child2[i] = np.random.randint(4)
            
    return child1, child2

def cross_over_with_digit(parent1, parent2):
    children={0:np.random.randint(gene_options, size=gene_size), 1:np.random.randint(gene_options, size=gene_size)}
    for child in range(2):
        for gene in range(gene_size):
            if(np.random.uniform() < mutation_chance):
                children[child][gene] = np.random.randint(4)
                
            else:
                if np.random.uniform() < 0.5:
                    children[child][gene] = parent1[gene]
                
                else:
                    parent2
                    
    return children[0], children[1]

def play_agent(agent, episode, env):
    total_reward=0
    for ep in range(episode):
        state=env.reset()
        while True:
            action=agent[state]
            next_state, reward, done, info=env.step(action)
            if done:
                total_reward+=reward
                break
            state = next_state
            
    return total_reward/episode
    
population = [np.random.randint(gene_options, size=gene_size) for _ in range(population_size)]
best_scores_with_peaces_crossover=[]

for n in range(generation_number):
    fitness_scores=[]
    for individual in population:
        fitness_scores.append(play_agent(individual, n_episode, env))
    best_individual=np.max(fitness_scores)
    best_scores_with_peaces_crossover.append(best_individual)
    population_ranks=list(reversed(np.argsort(fitness_scores)))
    elite_pop=[population[x] for x in population_ranks[:elite_pop_size]]
    select_probs=np.array(fitness_scores)/np.sum(fitness_scores)
    child_set=[]
    for generate in range(23):
        children=cross_over_with_pieces_of_parents(population[np.random.choice(range(population_size), p=select_probs)], population[np.random.choice(range(population_size), p=select_probs)])
        child_set.append(children[0])
        child_set.append(children[1])
    population=child_set+elite_pop
    
y1=best_scores_with_peaces_crossover
x=np.arange(20)

plt.plot(x,y1)
plt.title("Genetik Sonuclar")
plt.show()