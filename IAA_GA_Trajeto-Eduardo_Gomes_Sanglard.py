import random, math, os
import imageio.v2 as imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Define the number of cities
NUM_CITIES = 100

# Define the size of the population
POPULATION_SIZE = 200

# Define the number of generations
NUM_GENERATIONS = 2500

# Define the mutation rate
MUTATION_RATE = 0.01

# Define the crossover rate
CROSSOVER_RATE = 0.6

# Define the tournament size
TOURNAMENT_SIZE = 5

# Define the elitism factor
ELITISM_FACTOR = 0.5

# Define Checkpoint Frequency for showing the best route
CHECKPOINT_FREQUENCE = 250

# Min and max values for city coordinates
MIN_VAL = 1
MAX_VAL = 100

# Set Random Generator Seed
random.seed(1998)

# Define the cities with their locations (x and y)
CITIES = [(random.sample(range(MIN_VAL, MAX_VAL + 1), 2)) for _ in range(NUM_CITIES)]

def fitness(individual) -> float:
    """
    Define the fitness function
    Receives an individual and returns the fitness value
    """
    total_distance = 0
    for i in range(len(individual) - 2):
        city1 = individual[i]
        city2 = individual[i + 1]
        distance = math.sqrt((city2[0] - city1[0]) ** 2 + (city2[1] - city1[1]) ** 2)
        total_distance += distance
    return 1 / total_distance

def initialize_population() -> list:
    """
    Initialize the population
    """
    population = []
    for i in range(POPULATION_SIZE):
        individual = random.sample(CITIES, NUM_CITIES)
        population.append(individual)
    return population

def selection(population) -> list:
    """
    Define the selection function
    Receives the popluations and returns the selected individual by tournament selection
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    best_individual = max(tournament, key=fitness)
    return best_individual

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:

        child1 = []
        child2 = []
        crossover_point = random.randint(1, len(parent1) - 1)

        child1[:crossover_point] = parent1[:crossover_point]
        new_genes = [gene for gene in parent2 if gene not in child1]
        child1 = child1 + new_genes

        child2[:crossover_point] = parent2[:crossover_point]
        new_genes = [gene for gene in parent1 if gene not in child2]
        child2 = child2 + new_genes


    else:
        child1 = parent1
        child2 = parent2

    if (len(child1) != NUM_CITIES or len(child2) != NUM_CITIES):
        logging.debug('CROSSOVER returned a child with a different number of genes')
    
    return child1, child2

def mutation(individual):
    """
    Define the mutation function
    Receives the individual and returns the mutated individual
    """
    for i in range(len(individual) - 1):
        if random.random() < MUTATION_RATE:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual
    
def plot_route(best_route: list, img_name = 'Route', show_plot = False) -> None:

    """
    Plot Route
    """

    img_file_path =  f'Data\{img_name}.png'

    # Plot best route
    fig, ax = plt.subplots()

    x_axis = [city[0] for city in best_route] 
    y_axis = [city[1] for city in best_route] 

    # Plot the dots
    ax.scatter(x_axis, y_axis, color='blue')

    for i_city in range(len(best_route)):
        
        if i_city+1 < len(best_route):

            from_city = best_route[i_city]
            to_city = best_route[i_city+1]

            ax.plot([from_city[0], to_city[0]], [from_city[1], to_city[1]], color='red', linewidth=0.5)

    plt.title(img_name)

    if os.path.exists(img_file_path):
        os.remove(img_file_path)

    plt.savefig(img_file_path)

    if show_plot:
        plt.show()
    
    plt.close()

def create_gif(folder: str) -> None:
    """Create a gif from the images in the folder"""
    frames = []
    files = [file for file in os.listdir(folder) if file.endswith('.png') and file.startswith('Generation')]
    files = sorted([file for file in files],key = lambda x: int(x.split(' ')[1].split('.')[0]))
    files.append('/Best Route.png')
    for filename in files:
        frames.append(imageio.imread(folder + filename))
    imageio.mimsave(folder + 'GA_Evolution.gif', frames, fps=2)

def delete_files(folder: str) -> None:
    """Delete all files in a folder"""
    files = [file for file in os.listdir(folder) if file.endswith('.png')]
    for filename in files:
        os.remove(folder + filename)

def genetic_algorithm() -> None:

    """
    Define the genetic algorithm function
    """

    population = initialize_population()
    fitness_history = []

    for generation in tqdm(range(NUM_GENERATIONS)):

        sorted_population = sorted(population, key=fitness, reverse=True)
        elite_size = int(ELITISM_FACTOR * POPULATION_SIZE)
        elites = sorted_population[:elite_size]

        new_population = elites
        while len(new_population) < POPULATION_SIZE:
            
            parent1 = selection(elites)
            parent2 = selection(elites)

            # Verification made to ensure that the same individual will not be selected twice
            while parent1 == parent2:
                parent2 = selection(population)

            parent1, parent2 = crossover(parent1, parent2)

            new_population.append(parent1)
            new_population.append(parent2)

        for individual in new_population:
            individual = mutation(individual)
        
        population = new_population

        if len(population) != POPULATION_SIZE:
            logging.warning('Population is shrinking!')

        best_individual = max(population, key=fitness)
        fitness_history.append(fitness(best_individual))
        
        if generation % CHECKPOINT_FREQUENCE == 0:
            plot_route(best_individual, img_name=f'Generation {generation}')


    best_individual = max(population, key=fitness)
    return best_individual, fitness_history

if __name__ == '__main__':
    
    delete_files('Data/')

    # Run the genetic algorithm
    best_route, fitness_history = genetic_algorithm()

    # Print the best route
    plot_route(best_route, img_name='Best Route', show_plot=True)

    # Plot Fitness by generations
    plt.plot(fitness_history)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title('Fitness by generations')
    plt.show()

    create_gif('Data/')
