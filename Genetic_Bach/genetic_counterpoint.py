import numpy as np
from scipy import stats
import random
import csv
import os
from music21 import pitch as pt, note as no, stream, duration as du, converter, tempo, key, converter

CONST = 1e-12

def read_csv(filename):

    """
        Arg: 
            filename: path of the file
        Return:
            data: list
    """ 

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip the header row
        data = np.zeros((0,2))
        
        for row in reader:
            data = np.vstack((data, [float(row[2]), float(row[3])]))
            data = data.tolist()
       
    return data


def pairwise_dist(x, y):

        np.random.seed(1)

        """
            Args:
                x: N x D numpy array
                y: M x D numpy array
            Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between x[i, :] and y[j, :]
        """  

        dist = np.sqrt(
            np.sum(y**2, axis=1)[np.newaxis, :] + np.sum(x**2, axis=1)[:, np.newaxis] - 2*np.matmul(x, y.T) + CONST
            )        
        
        return dist


def get_population(dir):
    
    """
        Arg: 
            dir : path of the population csv files
        Return:
            population: list
    """ 

    files = []
    for file in os.listdir(dir):
        if file.endswith('.csv'):
            files.append(os.path.join(dir, file))
    
    population = []
    for file in files:
        data = read_csv(file)
        population.append(data)
    
    return population


def reshape_arr(arr1: np.ndarray, arr2: np.ndarray):

    """
        Args:
            arr1: N x D numpy array
            arr2: M x D numpy array
        Return:
            arr2: min(N, M) x D numpy array
    """  
    
    arr1 = np.ravel(arr1)
    arr2 = np.ravel(arr2)

    # Pad zeros to arr2 if it is shorter than arr1
    if len(arr2) < len(arr1):
        arr2 = np.pad(arr2, (0, len(arr1) - len(arr2)), mode='constant')

    # Truncate arr2 if it is longer than arr1
    if len(arr2) > len(arr1):
        arr2 = arr2[:len(arr1)]
        
    arr2 = arr2.reshape(-1,1)
    return arr2


def linear_normalize(input: np.ndarray):

    """
        Normalize a matrix to [0, 1].
            Arg:
                input: a NumPy array of shape (m, n), where m is the number of rows and n is the number of columns
            Return:
                normalized_arr: a NumPy array of shape (m, n), where each column has zero mean and unit variance
    """
    
    inpt_max = np.max(input, axis=0)
    inpt_min = np.min(input, axis=0)
    
    normalized_arr = (input - inpt_min) / (inpt_max - inpt_min + CONST)
    return(normalized_arr)


def stat_zscore(input: np.ndarray):

    """
        Normalize a matrix using Z-score normalization, and handle columns with zero standard deviation.
            Arg:
                input: a NumPy array of shape (m, n), where m is the number of rows and n is the number of columns
            Return:
                zscore_matrix: a NumPy array of shape (m, n), where each column has zero mean and unit variance
    """
    
    zscore_matrix = stats.zscore(input)
    zscore_matrix = np.nan_to_num(zscore_matrix, nan=0)

    return zscore_matrix


def fitness(input: np.ndarray, population, survial_rate):

    """
        Args: 
            input : N x 2 numpy array
            population: list
            survival_rate: percentage of the population remain
        Return:
            survival population list
    """ 

    if input.shape[1] != 2:
        raise TypeError("input should be N x 2")

    input_pitch, input_dur = np.split(input, 2, axis=1)
    scores = []
    new_pop = []
    dist_pitch = []
    dist_dur = []

    for i in range(0, len(population)):
        
        curr_pop = np.array(population[i]).reshape(len(population[i]),2)
        curr_pop_pitch, curr_pop_dur = np.split(curr_pop, 2, axis=1)
        
        curr_pop_pitch = reshape_arr(input_pitch, curr_pop_pitch)  
        curr_pop_dur = reshape_arr(input_dur, curr_pop_dur)
        
        dist_p = pairwise_dist(input_pitch, curr_pop_pitch)
        dist_p = np.mean(np.diag(dist_p))
        dist_pitch.append(dist_p)

        dist_d = pairwise_dist(input_dur, curr_pop_dur)
        dist_d = np.mean(np.diag(dist_d))
        dist_dur.append(dist_d)

    distpitch = np.array(dist_pitch)
    distpitch = linear_normalize(distpitch)
    distdur = np.array(dist_dur) 
    distdur = linear_normalize(dist_dur)

    scores = 0.7 * distpitch + 0.3 * distdur
    sorted_indices = np.argsort(scores)
    idx = sorted_indices[:int(len(population) * survial_rate)+1].tolist()
    
    for i in idx:
        new_pop.append(population[i])
    
    population = new_pop

    return population


def crossover(population):
    
    """
        Single point crossover function:
            Arg: 
                population: list
            Return:
                population: list, append offsprint to original population
    """ 
    
    for _ in range(round(len(population)/5)):
        
        #parent1 = random.choice(population)
        parent1 = population[random.randint(0,2)]
        parent2 = random.choice([p for p in population if p != parent1])
        
        crossover_point = random.randint(round(min(len(parent1), len(parent2))/4), min(len(parent1), len(parent2)))
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        population.append(offspring1)
        population.append(offspring2)
    
    return population


def mutate(population, probability):
    
    """
        Args: 
            population: list
            probability: probability of the member in population to mutate
        Return:
            population: list, append offsprint to original population
    """ 

    pitch_values = [-2.0, 2.0, -4.0, 4.0, -7.0, 7.0]
    pitch_weights = [0.05, 0.05, 0.2, 0.2, 0.25, 0.25]
    
    dur_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    dur_weights = [0.01, 0.15, 0.68, 0.15, 0.01]

    for i in range(len(population)):

        # check if a mutation should occur for this individual
        if random.random() < probability:

            # randomly select one or two elements to mutate
            idx = random.randint(0, len(population[i])-1)
            # change the element to a random value
            population[i][idx] = [(population[i][idx][0] + random.choices(pitch_values, weights=pitch_weights, k=1)[0]), 
                                  random.choices(dur_values, weights=dur_weights, k=1)[0]]
            
            if random.random() < 0.5:
                idx2 = random.randint(0, len(population[i])-1)
                # change the second element to a random value
                population[i][idx2] = [(population[i][idx2][0] + random.choices(pitch_values, weights=pitch_weights, k=1)[0]), 
                                       random.choices(dur_values, weights=dur_weights, k=1)[0]]

    return population


def check_total_dur(output):

    """
        Arg: 
            output: list of melody sequence
        Return:
            selected_output: list of melody sequence that total duration <= 16
    """ 
    
    best = np.array(output).reshape(len(output),2)
    sum_dur = 0
    selected_indices = []

    for i, value in enumerate(best[:, 1]):
        if sum_dur + value > 16:
            break
        sum_dur += value
        selected_indices.append(i)

    selected_output = best[:len(selected_indices), :].tolist()

    return selected_output 


def run_genetic(MAX_ITER, SURVIVAL_RATE, MUTATE_RATE, INPUT, DIR):

    """
        Args: 
            MAX_ITER: the maximum number of iterations
            SURVIVAL_RATE: percentage of population to survival after crossover
            MUTATE_RATE: percentage of population of mutation
            INPUT: input numpy array
            DIR: path of population folder
        Return:
            best_individual: the first element in the population list after running the max number of iteration (without the elements when dur >= 16).
    """ 
    
    pop = get_population(DIR)
    
    for _ in range(0, MAX_ITER):
        pop = fitness(INPUT, pop, SURVIVAL_RATE)
        crossover(pop)
        mutate(pop, MUTATE_RATE)
        #print(len(pop))

    best_individual = check_total_dur(pop[0])
    '''best_individual_2 = check_total_dur(pop[1])
    best_individual_3 = check_total_dur(pop[2])

    # Keep selecting new individuals until we have three distinct ones
    while best_individual_2 == best_individual_1 or best_individual_2 == best_individual_3:
        best_individual_2 = check_total_dur(np.random.choice(pop))

    while best_individual_3 == best_individual_1 or best_individual_3 == best_individual_2:
        best_individual_3 = check_total_dur(np.random.choice(pop))
        
    return best_individual_1, best_individual_2, best_individual_3'''
    return best_individual


def write_midi(arr_list, output_dir):

    """
        Args: 
            arr_list: list of notes [pitch, dur]
            output_dir: output path for midi file
    """ 

    arr = np.array(arr_list).reshape(len(arr_list),2)
    
    notes = [int(note[0]) for note in arr]
    durations = [float(note[1]) for note in arr]

    mstream = stream.Stream()

    for i, nt in enumerate(notes):   
        
        duration = durations[i]
        pitch_obj = pt.Pitch(nt)
        
        n = no.Note()
        n.pitch = pitch_obj
        n.duration = du.Duration(duration/2)
        
        ks = key.Key('C')
        mstream.insert(0, ks)
        mm = tempo.MetronomeMark(number=40)
        mstream.insert(0, mm)

        mstream.append(n)
    
    quarterLengthDivisors = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    mstream = mstream.quantize(quarterLengthDivisors)
    mstream.write('midi', fp=output_dir)
    #mstream.show()


def read_midi(input_dir):

    """
        Arg: 
            input_dir: path to midi file to read
        Return:
            note_dur_array: N by 2 numpy array [pitch, dur] of the midi file
    """ 
    
    midi_stream = converter.parse(input_dir)

    note_dur_array = []

    for note in midi_stream.flat.notes:
        
        pitch = note.pitch.midi
        dur = note.duration.quarterLength
        note_dur_array.append([pitch, dur])

    note_dur_array = np.array(note_dur_array)

    return note_dur_array



for i in range(1,9):

    input_dir = f'/Users/annie/Desktop/2023 Spring/MUSI 7100/week12/input_test/test0{i}.mid'
    note_dur_array = read_midi(input_dir)
    input_list = note_dur_array.tolist()
    print(input_list)

    pop_dir = '/Users/annie/Desktop/Genetic_Bach/midiCmaj/4_4/DATA'
    output_dir = f'/Users/annie/Desktop/2023 Spring/MUSI 7100/week12/output/test0{i}_output.mid'
    output_quant_dir = f'/Users/annie/Desktop/2023 Spring/MUSI 7100/week12/output_quant/test0{i}_output_quant.mid'

    best_individual = run_genetic(5, 0.25, 0.02, note_dur_array, pop_dir)
    result = input_list + best_individual
    print(result)
    
    write_midi(result, output_dir)