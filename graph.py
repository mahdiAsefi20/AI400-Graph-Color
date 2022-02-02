import random
import math
import numpy as np
import matplotlib.pyplot as plt

COLOR = ['RED' , 'BLUE' , 'WHITE', 'YELLOW', 'GREEN' ]


class Node:
    def __init__(self, ID , node_list = [] , color = None):
        self.__id = ID
        self.__connections = node_list.copy()
        self.__color = color

    def get_id(self):
        return self.__id

    def neighbours(self):
        return self.__connections

    def get_degree(self):
        return len(self.__connections)

    def create_connection(self, node):

        # if node not in self.__connections:

        if (node not in self.__connections) and (node is not self):
            self.__connections.append(node)
            node.create_connection(self)


    def destroy_connection(self, id):
        
        for elm in self.__connections:
            if elm.get_id() == id:
                
                self.__connections.remove(elm)
                
                return True

        return False

    def colorize(self , color):
        self.__color = color

    def get_color(self):
        return self.__color



    def number_of_conflicts(self):
      n_conflicts = 0
      for n in self.neighbours():
        if self.get_color() == n.get_color():
          n_conflicts = n_conflicts + 1
      
      return n_conflicts


    def __repr__(self):
        return  str(self.__id)


########################################################################################################
class Graph:

    def __init__(self, name = 'default' , node_list = []):
        self.nodes = node_list
        self.name = name


    def is_coloring_valid(self):

        for node in self.nodes:

            my_color = node.get_color()

            for neighbour in node.neighbours():
                
                if my_color == neighbour.get_color():
                    return False

        return True


    
    def get_node_list(self):
      return self.nodes




    @staticmethod
    def read_file(pth):
        with open(pth , 'r') as file:
            num_of_nodes = int( file.readline() )
            
            # node_list = [Node(name) for name in range(1 , num_of_nodes + 1)]    # list comperhension
            
            node_list = { str(name): Node(name) for name in range(1 , num_of_nodes + 1)}

            
            for index in range(1 , num_of_nodes + 1 ):
                file_line = file.readline()
                current_node_list = file_line.split(' ')[:-1]

                
                for node_key in current_node_list:
                    if (node_key == '-1'):
                        break
                    
                    node_list[str(index)].create_connection(
                        node_list[node_key]
                    )
                    
        

        return Graph(node_list = list(node_list.values()))  







########################################################################################################

class GeneticAlgorithm:

  def __init__(self, Graph, n_population, epoch, mutate_probability):
    self.graph = Graph
    self.n_pop = n_population
    self.epoch = epoch
    self.mutate_p = mutate_probability

  def initial_population(self):
    population = []
    n_nodes = len(self.graph.get_node_list())
    for c in range(self.n_pop):
      choromosome = []
      for i in range(n_nodes):
        r = random.randint(0, len(COLOR)-1)
        choromosome.append(COLOR[r])
      population.append(choromosome)

    return population


  def fitness(self, choromosome):

    n_conflict = 0

    nodes = self.graph.get_node_list()
    for node in nodes:
      node.colorize(choromosome[nodes.index(node)])
    
    for node in nodes:
      n_conflict = n_conflict + node.number_of_conflicts()

    f = 1 / (math.exp(n_conflict))
    return f


  def crossover_onePoint(self, parent1, parent2):
    child = []
    point = random.randint(0,len(parent1)-1)
    child = parent1[:point]
    child[point:] = parent2[point:]
    return child


  def crossover_twoPoint(self, parent1, parent2):
    child = []
    point1 = random.randint(0,len(parent1)/2)
    print(point1)
    point2 =  random.randint(point1 + 1 ,len(parent1)-1)
    print(point2)
    child = parent1[:point1]
    child[point1:point2] = parent2[point1:point2]
    child[point2:] = parent1[point2:]
    return child

  def mutate(self, child):
    index = random.randint(0, len(child)-1)
    r = random.randint(0, len(COLOR)-1)
    child[index] = COLOR[r]
    return child

  def random_selection(self, population):
    fitnessList = [] 
    for chromosome in population:
      f = self.fitness(chromosome)
      fitnessList.append(f)

    population_fitness = sum(fitnessList)

    probabilities = []
    for fitness in fitnessList:
      probability = fitness / population_fitness
      probabilities.append(probability)

    Index_chromosome1 = probabilities.index(np.random.choice(probabilities, p = probabilities))
    Index_chromosome2 = probabilities.index(np.random.choice(probabilities, p = probabilities))

    Selected_chromosome1 = population[Index_chromosome1]
    Selected_chromosome2 = population[Index_chromosome2]

    return Selected_chromosome1, Selected_chromosome2, fitnessList


  def population_info(self, fitnessList):

    avg = sum(fitnessList)/len(fitnessList)
    best = max(fitnessList)
    return avg, best
  


  def train(self, crossover):
    Population = self.initial_population()
    Averages = []
    Bests = []

    e = 0
    while e < self.epoch:
      new_Population = []
      for n in range(self.n_pop):
        Chromosome1,Chromosome2, FitnessList = self.random_selection(Population)
        new_Chromosom = crossover(Chromosome1,Chromosome2)
        p = random.random()
        if(p < self.mutate_p):
          new_Chromosom = self.mutate(new_Chromosom)
        new_Population.append(new_Chromosom)
      Population = new_Population
      print("Epoch: ",e)
      e = e + 1
      avg, best = self.population_info(FitnessList)
      print("Average Fitness: ", avg)
      Averages.append(avg)
      print("Best Fitness: ", best)
      Bests.append(best)
      print("Population",len(FitnessList))
      if best == 1:
        solution = Population[FitnessList.index(best)]
        print(solution)
        break
      print("              ---------------------------------------------------------")
    
    return Averages, Bests

################################################################################################

my_graph = Graph.read_file('/content/sample-graph.gp') #graph file path
genetic = GeneticAlgorithm( my_graph, n_population= 3, epoch= 100, mutate_probability= 0.2)
Averages, Bests = genetic.train(genetic.crossover_onePoint)
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(Averages)
plt.title("Averages")
plt.subplot(122)
plt.plot(Bests)
plt.title("Best")
plt.show()
