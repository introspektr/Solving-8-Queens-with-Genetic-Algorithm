# Cody Jackson
# CS 441 Programming Assignment #2
# 5/18/25
# Solving 8-Queens Puzzle with Genetic Algorithm

import random
import time
from typing import List, Tuple

# ── hyperparameters ───────────────────────────────────────────────────────────
QUEENS = 8
POPULATION_SIZE = 100
MAX_GENERATIONS = 3000
MUTATION_RATE = 0.05  # 5% chance
MAX_FITNESS = 28      # Maximum fitness (28 non-attacking pairs)

# ── genetic algorithm functions ───────────────────────────────────────────────
def random_permutation() -> List[int]:
    """Create a random 8-Queens solution (chromosome)."""
    individual = list(range(QUEENS))
    random.shuffle(individual)
    return individual

def fitness_function(individual: List[int]) -> int:
    """
    Calculate fitness: 28 - (number of attacking pairs)
    Higher value is better, 28 is perfect solution.
    """
    attacks = 0
    # Check for attacking pairs
    for column1 in range(QUEENS):
        for column2 in range(column1 + 1, QUEENS):
            row1, row2 = individual[column1], individual[column2]
            # Queens attack if in same row or on same diagonal
            if row1 == row2 or abs(row1 - row2) == abs(column1 - column2):
                attacks += 1
    return MAX_FITNESS - attacks

def select_parent(population: List[List[int]], fitnesses: List[int]) -> List[int]:
    """Select parent using roulette wheel selection."""
    # Calculate total fitness
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        # If total fitness is zero, select randomly
        return random.choice(population)
    
    # Roulette wheel selection
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, individual in enumerate(population):
        current += fitnesses[i]
        if current > pick:
            return individual

def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Single-point crossover: combine genetic material from two parents."""
    # Select random crossover point
    point = random.randint(1, QUEENS - 2)
    # Create children by combining parts of parents
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual: List[int]) -> List[int]:
    """Randomly mutate by swapping two positions."""
    if random.random() < MUTATION_RATE:
        # Select two random positions and swap them
        pos1, pos2 = random.sample(range(QUEENS), 2)
        individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
    return individual

def display_board(permutation: List[int]) -> None:
    """Display the 8-Queens board with the given permutation."""
    board = []
    for row in range(QUEENS):
        line = ""
        for column in range(QUEENS):
            if permutation[column] == row:
                line += "Q "
            else:
                line += ". "
        board.append(line)
    print("\n" + "-" * 19)
    for row in board:
        print("| " + row + "|")
    print("-" * 19)

# ── main algorithm ──────────────────────────────────────────────────────────
def main():
    # Print header
    print("=" * 80)
    print(f"8-Queens Genetic Algorithm")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Maximum Generations: {MAX_GENERATIONS}")
    print(f"Mutation Rate: {MUTATION_RATE * 100}%")
    print("=" * 80)
    
    # Generate initial population
    population = [random_permutation() for _ in range(POPULATION_SIZE)]
    
    # Compute fitnesses for initial population
    fitnesses = [fitness_function(individual) for individual in population]

    print("\nSTARTING GENETIC ALGORITHM...")
    
    # Table header for fitness data (printed for every generation)
    print("\nFITNESS DATA (BEST & AVERAGE FITNESS) FOR EACH GENERATION:")
    print("-" * 60)
    print(f"{'Generation':<12}{'Best Fitness':<16}{'Average Fitness':<20}")
    print("-" * 60)
    
    start_time = time.time()
    solution_found = False
    
    # Define generations to collect a sample chromosome (0, 1–5, every 25th (up to 100), and every 50th (after 100)
    sample_gens = [0] + list(range(1, 6)) + list(range(25, 101, 25)) + list(range(150, MAX_GENERATIONS + 1, 50))
    
    # List to collect sample chromosomes (each entry is a tuple (gen, chromosome, fitness))
    sample_chromosomes = []
    
    # Main loop
    for generation in range(MAX_GENERATIONS + 1):
        # Calculate fitness for each individual
        fitnesses = [fitness_function(individual) for individual in population]
        
        # Find highest fitness in this generation
        highest_fitness_index = fitnesses.index(max(fitnesses))
        highest_fitness = fitnesses[highest_fitness_index]
        highest_fitness_individual = population[highest_fitness_index]
        
        # Calculate average fitness
        avg_fitness = sum(fitnesses) / len(fitnesses)
        
        # Print fitness data (best & average) for every generation
        print(f"{generation:<12}{highest_fitness:<16}{avg_fitness:.2f}")
        
        # Check if we've found a solution
        if highest_fitness == MAX_FITNESS and not solution_found:
            solution_found = True
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 60)
            print(f"SOLUTION FOUND IN GENERATION {generation}!")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Solution chromosome: {highest_fitness_individual}")
            display_board(highest_fitness_individual)
            print("=" * 60)
            # Sample the chromosome stored at population[7].
            sample_chromosome = population[7]
            sample_fitness = fitness_function(sample_chromosome)
            sample_chromosomes.append((generation, sample_chromosome, sample_fitness))
            break
        
        # Sample the chromosome stored at population[7].
        if generation in sample_gens:
            sample_chromosome = population[7]
            sample_fitness = fitness_function(sample_chromosome)
            sample_chromosomes.append((generation, sample_chromosome, sample_fitness))
        # -------------------------------------------------- #
        
        # Create new generation
        new_population = []
        
        # Create list of (fitness, individual) pairs
        fitness_pairs = []
        for i in range(len(population)):
            fitness_pairs.append((fitnesses[i], population[i]))
        
        # Sort in descending order by fitness
        fitness_pairs.sort(reverse=True)
        
        # Elitism: keep the best 2 individuals
        elites = [individual for fitness, individual in fitness_pairs[:2]]
        new_population.extend(elites)
        
        # Fill the rest of the population with children
        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            
            # Create children through crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Apply mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            new_population.append(child2)
        
        # Ensure population size is exactly POPULATION_SIZE
        population = new_population[:POPULATION_SIZE]
    
    # If we didn't find a solution, show the best one found
    if not solution_found:
        print("\n" + "=" * 60)
        print("FINAL RESULTS (NO PERFECT SOLUTION FOUND):")
        final_fitnesses = [fitness_function(individual) for individual in population]
        best_fitness_index = final_fitnesses.index(max(final_fitnesses))
        best_fitness = population[best_fitness_index]
        print(f"Best Fitness Achieved: {final_fitnesses[best_fitness_index]}")
        print(f"Best Chromosome: {best_fitness}")
        display_board(best_fitness)
        print("=" * 60)
    
    # Always show the final population statistics
    print("\nFINAL POPULATION STATISTICS:")
    print("-" * 60)
    final_fitnesses = [fitness_function(individual) for individual in population]
    avg_final_fitness = sum(final_fitnesses) / len(final_fitnesses)
    print(f"Average Fitness: {avg_final_fitness:.2f}")
    print(f"Best Fitness: {max(final_fitnesses)}")
    print("-" * 60)
    
    # Display collected sample chromosomes (boardstates) (one per sample generation)
    print("\nSAMPLE CHROMOSOMES COLLECTED DURING THIS RUN FROM POPULATION[7]):")
    print("=" * 60)
    for (generation, chromosome, fitness) in sample_chromosomes:
        print(f"\nSample (Generation {generation}):")
        print(f"Chromosome: {chromosome}")
        print(f"Fitness: {fitness}")
        display_board(chromosome)
        print("-" * 60)

if __name__ == "__main__":
    main()