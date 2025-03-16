import torch
import torch.optim as optim
from copy import deepcopy
import random


# Genetic Algorithm Parameters
POPULATION_SIZE = 8
NUM_GENERATIONS = 1
MUTATION_RATE = 0.1
ELITE_SIZE = 2

# Hyperparameter search space
OPTIMIZERS = ['Adam', 'SGD', 'RMSprop']
LEARNING_RATES = [0.1, 0.01, 0.001]
WEIGHT_DECAYS = [0.0, 0.0001]
BATCH_SIZES = [32, 64]

TOURNAMENT_SIZE = 3             # Number of candidates per tournament
EARLY_STOPPING_PATIENCE = 5     # Generations without improvement to trigger stop
ENFORCE_UNIQUE_INDIVIDUALS = True  # Prevent duplicate individuals

def create_individual():
    return {
        'optimizer': random.choice(OPTIMIZERS),
        'lr': random.choice(LEARNING_RATES),
        'weight_decay': random.choice(WEIGHT_DECAYS),
        'batch_size': random.choice(BATCH_SIZES)
    }

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual):
    mutated = deepcopy(individual)
    if random.random() < MUTATION_RATE:
        mutated['optimizer'] = random.choice(OPTIMIZERS)
    if random.random() < MUTATION_RATE:
        mutated['lr'] = random.choice(LEARNING_RATES)
    if random.random() < MUTATION_RATE:
        mutated['weight_decay'] = random.choice(WEIGHT_DECAYS)
    if random.random() < MUTATION_RATE:
        mutated['batch_size'] = random.choice(BATCH_SIZES)
    return mutated

def evaluate_individual(train_subset, val_subset, individual, model_class, device, criterion, num_epochs=10):
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=individual['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=individual['batch_size'], shuffle=False
    )
    
    model = model_class().to(device)
    
    if individual['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                             lr=individual['lr'],
                             weight_decay=individual['weight_decay'])
    elif individual['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), 
                            lr=individual['lr'],
                            momentum=0.9,
                            weight_decay=individual['weight_decay'])
    else:
        optimizer = optim.RMSprop(model.parameters(),
                                lr=individual['lr'],
                                weight_decay=individual['weight_decay'])
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

'''
def run_genetic_algorithm(model_class, train_subset, val_subset, device, criterion):
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_fitness = -float('inf')

    for generation in range(NUM_GENERATIONS):
        # Parallel evaluation (example using ThreadPool)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(evaluate_individual, ...) for ind in population]
            fitness = [f.result() for f in futures]

        # Track best individual
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_individual = deepcopy(population[current_best_idx])

        # Selection (tournament selection)
        parents = tournament_selection(population, fitness, num_parents=...)

        # Crossover + mutation with probability
        children = []
        for _ in range(POPULATION_SIZE - ELITE_SIZE):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_PROB:  # Global mutation probability
                child = mutate(child)
            children.append(child)

        # New population (elite + children)
        elite = sorted(zip(population, fitness), key=lambda x: -x[1])[:ELITE_SIZE]
        population = [ind for ind, _ in elite] + children

    return best_individual  # No need for final re-evaluation'''

def run_genetic_algorithm(model_class, train_subset, val_subset, device, criterion):
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_fitness = -float('inf')
    stagnation_counter = 0  # For early stopping

    for generation in range(NUM_GENERATIONS):
        print(f"\n=== Generation {generation+1}/{NUM_GENERATIONS} ===")
        
        # Evaluation
        fitness = []
        for idx, individual in enumerate(population):
            print(f"Evaluating individual {idx+1}: {individual}")
            acc = evaluate_individual(train_subset, val_subset, individual, model_class, device, criterion)
            fitness.append(acc)
            print(f"Validation Accuracy: {acc:.2f}%")

        # Track best individual (re-does evaluation each gen., could save and exclude elite individuals to speed-up)
        current_best = max(fitness)
        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = deepcopy(population[fitness.index(current_best)])
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Early stopping
        if stagnation_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at generation {generation+1}")
            break

        # Tournament selection for parents
        parents = []
        for _ in range(POPULATION_SIZE):  # Select enough parents for mating pool
            contenders = random.sample(list(enumerate(fitness)), TOURNAMENT_SIZE)
            contenders.sort(key=lambda x: x[1], reverse=True)
            parents.append(population[contenders[0][0]])

        # Create next generation
        new_population = []
        
        # Keep elite (unmodified)
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[:ELITE_SIZE]
        new_population.extend([deepcopy(population[i]) for i in elite_indices])

        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            # Crossover
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            
            # Mutation with probability
            child = mutate(child)  # Uses per-parameter MUTATION_RATE internally
            
            # Ensure unique child (optional)
            if ENFORCE_UNIQUE_INDIVIDUALS:
                while child in new_population:
                    child = mutate(child)
            
            new_population.append(child)

        population = new_population

    return best_individual