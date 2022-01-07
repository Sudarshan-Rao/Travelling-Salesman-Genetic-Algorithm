import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

#Calculate distance and assign fitness value for each instance in population
def calcDistance(instanceList, distanceList):
    fitness = 0
    pathDistance = 0
    for i in range(0, len(instanceList)):
        fromCity = instanceList[i]
        if i + 1 < len(instanceList):
            toCity = instanceList[i + 1]
        else:
            toCity = instanceList[0]

        pathDistance += distanceList[fromCity][toCity]
    if fitness==0:
        fitness = 1/(float(pathDistance)) #handle for divide by zero error
    return fitness

#Rank routes
def rankRoutes(populationList,distanceList):
    fitnessResults = {}
    for i in range(0,len(populationList)):
        fitnessResults[i] = calcDistance(populationList[i],distanceList)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def getroutes(ranked_list,current_gen):
    new_pop=[]
    sorted_index_list=[]

    for i in range(0, len(ranked_list)):
        sorted_index_list.append(ranked_list[i][0])

    for i in range(0, len(current_gen)):
        index = sorted_index_list[i]
        if index <= len(current_gen):
            new_pop.append(current_gen[index])
    return new_pop

#Choose Parents based on fitness values
def selectParents(rankedPopulation,eliteSize):
    parentsList = []
    df = pd.DataFrame(np.array(rankedPopulation), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        parentsList.append(rankedPopulation[i][0])

    for i in range(0, len(rankedPopulation) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(rankedPopulation)):
            if pick <= df.iat[i, 3]:
                parentsList.append(rankedPopulation[i][0])
                break
    return parentsList

def matingPool(population, selectionResults):
    mating_pool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        mating_pool.append(population[index])
    return mating_pool

def generateOffsprings(parentList,eliteSize):
    children = []
    length = len(parentList) - eliteSize
    randomParents = random.sample(parentList, len(parentList))

    #Add elites
    for i in range(0, eliteSize):
        children.append(parentList[i])

    for i in range(0, length):
        child = crossOver(randomParents[i], randomParents[len(parentList) - i - 1])
        children.append(child)
    #print("Children "+str(children))
    return children

#Crossover and generate offsprings
def crossOver(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    #Select two points
    pointA = int(random.random() * len(parent1))
    pointB = int(random.random() * len(parent1))

    A = min(pointA, pointB)
    B = max(pointA, pointB)

    for i in range(A, B):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    #print("Child "+str(child))
    return child

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    #print("Mutated individual "+str(individual))
    return individual

def mutatePopulation(population, mutationRate,elitesize):
    mutatedPop = []
    for i in range(0, elitesize):
        mutatedPop.append(population[i])

    for ind in range(elitesize, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    #print("Mutation "+str(mutatedPop))
    return mutatedPop

#make next generation
def nextGeneration(currentGen, distanceMatrix, mutationRate,eliteSize):
    popRanked = rankRoutes(currentGen,distanceMatrix) #rankRoutes(currentGen)
    selectionResults = selectParents(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = generateOffsprings(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate,eliteSize)
    bestRouteIndex = popRanked[0][0]
    bestRoute = currentGen[bestRouteIndex]
    bestDistance=1/popRanked[0][1]
    print("BestRouteIndex " +str(bestRouteIndex))
    print("BestRoute " +str(bestRoute))
    print("BestDistance " + str(bestDistance))
    print("CurrentGen: " +str(currentGen))
    print("NextGen: " +str(nextGeneration))
    return nextGeneration

def geneticAlgorithm():
    # Enter the number of cities n
    N = int(input("Please enter number of cities:\n"))
    # print(f'You entered {N}')
    # Create n*n matrix and enter matrix or generate random numbers
    intermediateMatrix = np.random.randint(0, 25, size=(N, N))
    CityDist = np.tril(intermediateMatrix) + np.tril(intermediateMatrix, -1).T
    for i in range(0, N):
        CityDist[i][i] = 0
    print(f'Distance Matrix: {CityDist}')

    # Ask the user to enter number of population and generate random populations
    pSize = int(input("Please enter number of populations:\n"))
    population = []
    for i in range(0, pSize):
        inst = random.sample(range(N), N)
        population.append(inst)
    print("Population " + str(population))

    progress = []
    progress.append(1/rankRoutes(population,CityDist)[0][1])

    #Ask the user to enter number of generations
    generations=int(input("Please enter number of generations:\n"))

    #Ask the user to enter elite size
    eliteSize=int(input("Please enter elite number of parents: \n"))

    mutationRate=float(input("Please Enter mutation rate: \n"))

    for i in range(0, generations):
        print("Generation " + str(i) + "\n")
        population = nextGeneration(population, CityDist, mutationRate, eliteSize)
        print("Progress of Distance :" + str(progress))
        progress.append(1/rankRoutes(population, CityDist)[0][1])

    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()

geneticAlgorithm()
