import operator
import random
import classifier

dataset = classifier.read()
pop_n = 100
gene_n = len(dataset[0])
max_generation = 1000
crossover_rate = 0.7
mutation_rate = 0.05

def filter(mask):
    subset = []
    for i in range(len(dataset)):
        subset.append(dataset[i][:3])
    for i in range(3, len(mask)):
        if mask[i] == 1:
            for j in range(len(dataset)):
                subset[j].append(dataset[j][i])
    return subset

def generate():
    chromosome = []
    for i in range(pop_n):
        mask = [1, 1, 1]
        for j in range(gene_n - 4):
            mask.append(random.randint(0, 1))
        mask.append(1)
        chromosome.append(mask)
    print "generated"
    return chromosome

def evaluate(chromosome):
    total_accuracy = []
    total = 0
    for chromo in chromosome:
        accuracy = classifier.cross_validation(filter(chromo))
        total += accuracy
        total_accuracy.append(accuracy)
    def normalize(num):
        return num / total
    fitness = map(normalize, total_accuracy)
    return zip(chromosome, total_accuracy, fitness)

def rank(chromosome):
    paired_population = evaluate(chromosome)
    ranked_population = sorted(paired_population, key = operator.itemgetter(-1), reverse = True)
    return ranked_population

def roulette(fitness):
    index = 0
    cumulative_fitness = 0.0
    r = random.random()
    for portion in fitness:
        cumulative_fitness += portion
        if cumulative_fitness > r:
            return index
        index += 1

def select_fittest(chromosome, fitness):
    (index1, index2) = (roulette(fitness), roulette(fitness))
    while index1 == index2:
        (index1, index2) = (roulette(fitness), roulette(fitness))
    return (chromosome[index1], chromosome[index2])

def cross_over(chromo1, chromo2):
    r = random.randint(3, gene_n - 1)
    return (chromo1[:r] + chromo2[r:], chromo2[:r] + chromo1[r:])

def mutate(chromo):
    for i in range(3, gene_n - 1):
        if random.random() < mutation_rate:
            chromo[i] = chromo[i] ^ 1

def breed(chromo1, chromo2):
    new_chromo1, new_chromo2 = [], []
    if random.random() < crossover_rate:
        (new_chromo1, new_chromo2) = cross_over(chromo1, chromo2)
    else:
        (new_chromo1, new_chromo2) = (chromo1, chromo2)
    mutate(new_chromo1)
    mutate(new_chromo2)
    return (new_chromo1, new_chromo2)

def next_generation(ranked_population):
    chromosome = [item[0] for item in ranked_population]
    fitness = [item[-1] for item in ranked_population]
    new_population = []
    new_population.extend(chromosome[:3])
    while len(new_population) < pop_n:
       (chromo1, chromo2) = select_fittest(chromosome, fitness)
       (chromo1, chromo2) = breed(chromo1, chromo2)
       new_population.append(chromo1)
       new_population.append(chromo2)
    return new_population

def main():
    chromosome = generate()
    generation = 0
    while generation < max_generation:
        print "Current generation: %d" % generation
        ranked_population = rank(chromosome)
        print "Current Best: {0:.2%}".format(ranked_population[0][1])
        chromosome = next_generation(ranked_population)
        generation += 1
    chromosome = rank(chromosome)
    fo = open("mask", "wb")
    for (mask, accuracy, fitness) in chromosome[:3]:
        for bit in mask[:-1]:
            fo.write(str(bit) + ' ')
        fo.write(str(mask[-1]) + '\n')
        fo.write(str(accuracy) + '\n')
    fo.close()

if __name__ == "__main__":
    main()
