import numpy as np
from scipy import stats
import random
import csv
import os
from music21 import pitch as pt, note as no, stream, duration as du, converter, tempo, key, converter, meter
import argparse
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import torch
import torch.nn as nn
import numpy as np
import mido
import matplotlib.pyplot as plt


CONST = 1e-12

parser = argparse.ArgumentParser()

parser.add_argument("--ip", default="127.0.0.1",
    help="The ip of the OSC server")
parser.add_argument("--portClient", type=int, default=5008,
    help="The port the OSC server is listening on")
parser.add_argument("--portServer", type=int, default=6006,
    help="The port to send to max")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.portClient)


has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "mps" if getattr(torch,'has_mps',False) \
    else "gpu" if torch.cuda.is_available() else "cpu"

print(device)


base_dir = '/Users/annie/Documents/Shimon-Counterpoint/cocotorch/'
base = '/Users/annie/Documents/Shimon-Counterpoint/'


# load training data
data = np.load(base_dir + 'Jsb16thSeparated.npz', encoding='bytes', allow_pickle=True)
all_tracks = []
for x in data.files:
    for y in data[x]:
        for i in range(-6, 6):
            all_tracks.append(y + i)
max_midi_pitch = -np.inf
min_midi_pitch = np.inf
for x in all_tracks:
    if x.max() > max_midi_pitch:
        max_midi_pitch = int(x.max())
    if x.min() < min_midi_pitch:
        min_midi_pitch = int(x.min())


I = 4 # number of voices
T = 32 # length of samples (32 = two 4/4 measures)
P = max_midi_pitch - min_midi_pitch +1 # number of different pitches
batch_size=24


# function for converting arrays of shape (T, 4) into midi files
# the input array has entries that are np.nan (representing a rest)
# of an integer between 0 and 127 inclusive

def piano_roll_to_midi(piece, track0_path=None):
    """
    piece is a an array of shape (T, 4) for some T.
    The (i,j)th entry of the array is the midi pitch of the jth voice at time i. It's an integer in range(128).
    track0_path is an optional path to a MIDI file that will be loaded as the first track of the output MIDI file.
    outputs a mido object mid that you can convert to a midi file by called its .save() method
    """
    piece = np.concatenate([piece, [[np.nan, np.nan, np.nan, np.nan]]], axis=0)

    bpm = 40
    microseconds_per_beat = 60 * 1000000 / bpm

    mid = mido.MidiFile()
    tracks = {'soprano': mido.MidiTrack(), 'alto': mido.MidiTrack(),
              'tenor': mido.MidiTrack(), 'bass': mido.MidiTrack()}
    past_pitches = {'soprano': np.nan, 'alto': np.nan,
                    'tenor': np.nan, 'bass': np.nan}
    delta_time = {'soprano': 0, 'alto': 0, 'tenor': 0, 'bass': 0}

    # create a track containing tempo data
    metatrack = mido.MidiTrack()
    metatrack.append(mido.MetaMessage('set_tempo',
                                      tempo=int(microseconds_per_beat), time=0))
    mid.tracks.append(metatrack)

    # load track 0 if provided
    #if track0_path is not None:
        #track0 = mido.MidiFile(track0_path).tracks[1]
        #mid.tracks.append(track0)

    # create the four voice tracks
    for voice in tracks:
        mid.tracks.append(tracks[voice])
        tracks[voice].append(mido.Message(
            'program_change', program=52, time=0))

    # add notes to the four voice tracks
    for i in range(len(piece)):
        pitches = {'soprano': piece[i, 0], 'alto': piece[i, 1],
                   'tenor': piece[i, 2], 'bass': piece[i, 3]}
        for voice in tracks:
            if np.isnan(past_pitches[voice]):
                past_pitches[voice] = None
            if np.isnan(pitches[voice]):
                pitches[voice] = None
            if pitches[voice] != past_pitches[voice]:
                if past_pitches[voice]:
                    tracks[voice].append(mido.Message('note_off', note=int(past_pitches[voice]),
                                                      velocity=80, time=delta_time[voice]))
                    delta_time[voice] = 0
                if pitches[voice]:
                    tracks[voice].append(mido.Message('note_on', note=int(pitches[voice]),
                                                      velocity=80, time=delta_time[voice]))
                    delta_time[voice] = 0
            past_pitches[voice] = pitches[voice]
            # 480 ticks per beat and each line of the array is a 16th note
            delta_time[voice] += 120

    return mid



class Chorale:
    """
    A class to store and manipulate an array self.arr that stores a chorale.
    """
    def __init__(self, arr, subtract_30=False):
        # arr is an array of shape (4, 32) with values in range(0, 57)
        self.arr = arr.copy()
        if subtract_30:
            self.arr -= 30
            
        # the one_hot representation of the array
        reshaped = self.arr.reshape(-1)
        self.one_hot = np.zeros((I*T, P))
        r = np.arange(I*T)
        self.one_hot[r, reshaped] = 1
        self.one_hot = self.one_hot.reshape(I, T, P)
        

    def to_image(self):
        # visualize the four tracks as a images
        soprano = self.one_hot[0].transpose()
        alto = self.one_hot[1].transpose()
        tenor = self.one_hot[2].transpose()
        bass = self.one_hot[3].transpose()
        
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(np.flip(soprano, axis=0), cmap='hot', interpolation='nearest')
        axs[0].set_title('soprano')
        axs[1].imshow(np.flip(alto, axis=0), cmap='hot', interpolation='nearest')
        axs[1].set_title('alto')
        axs[2].imshow(np.flip(tenor, axis=0), cmap='hot', interpolation='nearest')
        axs[2].set_title('tenor')
        axs[3].imshow(np.flip(bass, axis=0), cmap='hot', interpolation='nearest')
        axs[3].set_title('bass')
        fig.set_figheight(5)
        fig.set_figwidth(15)
        return fig, axs
    

    def save(self, filename, track0_path):
        # display an in-notebook widget for playing audio
        # saves the midi file as a file named name in base_dir/midi_files
        
        midi_arr = self.arr.transpose().copy()
        midi_arr += 30
        midi = piano_roll_to_midi(midi_arr, track0_path)
        midi.save(base + 'midi_files/' + filename)
        #play_midi('midi_files/' + filename)
        

    def elaborate_on_voices(self, voices, model):
        # voice is a set consisting of 0, 1, 2, or 3
        # create a mask consisting of the given voices
        # generate a chorale with the same voices as in voices
        mask = np.zeros((I, T))
        y = np.random.randint(P, size=(I, T))
        for i in voices:
            mask[i] = 1
            y[i] = self.arr[i].copy()
        return harmonize(y, mask, model)


    def score(self):
        consonance_dict = {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 0, 7: 1, 8: 1, 9: 1, 10: 0, 11: 0}
        consonance_score = 0
        for k in range(32):
            for i in range(4):
                for j in range(i):
                    consonance_score += consonance_dict[((self.arr[i, k] - self.arr[j, k]) % 12)]
        
        note_score = 0
        for i in range(4):
            for j in range(1, 32):
                if self.arr[i, j] != self.arr[i, j-1]:
                    note_score += 1
        return consonance_score, note_score
        
            
        
# harmonize a melody
def harmonize(y, C, model):
    """
    Generate an artificial Bach Chorale starting with y, and keeping the pitches where C==1.
    Here C is an array of shape (4, 32) whose entries are 0 and 1.
    The pitches outside of C are repeatedly resampled to generate new values.
    For example, to harmonize the soprano line, let y be random except y[0] contains the soprano line, let C[1:] be 0 and C[0] be 1.
    """
    model.eval()
    with torch.no_grad():
        x = y
        C2 = C.copy()
        num_steps = int(2*I*T)
        alpha_max = .999
        alpha_min = .001
        eta = 3/4
        for i in range(num_steps):
            p = np.maximum(alpha_min, alpha_max - i*(alpha_max-alpha_min)/(eta*num_steps))
            sampled_binaries = np.random.choice(2, size = C.shape, p=[p, 1-p])
            C2 += sampled_binaries
            C2[C==1] = 1
            x_cache = x
            x = model.pred(x, C2)
            x[C2==1] = x_cache[C2==1]
            C2 = C.copy()
        return x
    

def generate_random_chorale(model):
    """
    Calls harmonize with random initialization and C=0, and so generates a new sample that sounds like Bach.
    """
    y = np.random.randint(P, size=(I, T)).astype(int)
    C = np.zeros((I, T)).astype(int)
    return harmonize(y, C, model)



hidden_size = 32


class Unit(nn.Module):
    """
    Two convolution layers each followed by batchnorm and relu, plus a residual connection.
    """
    def __init__(self):
        super(Unit, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_size)
        self.relu2 = nn.ReLU()
        
        
    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.batchnorm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.batchnorm2(y)
        y = y + x
        y = self.relu2(y)
        return y
    

class Net(nn.Module):
    """
    A CNN that where you input a starter chorale and a mask and it outputs a prediction for the values
    in the starter chorale away from the mask that are most like the training data.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.initial_conv = nn.Conv2d(2*I, hidden_size, 3, padding=1)
        self.initial_batchnorm = nn.BatchNorm2d(hidden_size)
        self.initial_relu = nn.ReLU()
        self.unit1 = Unit()
        self.unit2 = Unit()
        self.unit3 = Unit()
        self.unit4 = Unit()
        self.unit5 = Unit()
        self.unit6 = Unit()
        self.unit7 = Unit()
        self.unit8 = Unit()
        self.unit9 = Unit()
        self.unit10 = Unit()
        self.unit11 = Unit()
        self.unit12 = Unit()
        self.unit13 = Unit()
        self.unit14 = Unit()
        self.unit15 = Unit()
        self.unit16 = Unit()
        self.affine = nn.Linear(hidden_size*T*P, I*T*P)
        

    def forward(self, x, C):
        # x is a tensor of shape (N, I, T, P)
        # C is a tensor of 0s and 1s of shape (N, I, T)
        # returns a tensor of shape (N, I, T, P)
        
        # get the number of batches
        N = x.shape[0]
        
        # tile the array C out of a tensor of shape (N, I, T, P)
        tiled_C = C.view(N, I, T, 1)
        tiled_C = tiled_C.repeat(1, 1, 1, P)
        
        # mask x and combine it with the mask to produce a tensor of shape (N, 2*I, T, P)
        y = torch.cat((tiled_C*x, tiled_C), dim=1)
        
        # apply the convolution and relu layers
        y = self.initial_conv(y)
        y = self.initial_batchnorm(y)
        y = self.initial_relu(y)
        y = self.unit1(y)
        y = self.unit2(y)
        y = self.unit3(y)
        y = self.unit4(y)
        y = self.unit5(y)
        y = self.unit6(y)
        y = self.unit7(y)
        y = self.unit8(y)
        y = self.unit9(y)
        y = self.unit10(y)
        y = self.unit11(y)
        y = self.unit12(y)
        y = self.unit13(y)
        y = self.unit14(y)
        y = self.unit15(y)
        y = self.unit16(y)
            
        # reshape before applying the fully connected layer
        y = y.view(N, hidden_size*T*P)
        y = self.affine(y)
        
        # reshape to (N, I, T, P)
        y = y.view(N, I, T, P)
                
        return y
    

    def pred(self, y, C):
        # y is an array of shape (I, T) with integer entries in [0, P)
        # C is an array of shape (I, T) consisting of 0s and 1s
        # the entries of y away from the support of C should be considered 'unknown'
        
        # x is shape (I, T, P) one-hot representation of y
        compressed = y.reshape(-1)
        x = np.zeros((I*T, P))
        r = np.arange(I*T)
        x[r, compressed] = 1
        x = x.reshape(I, T, P)
        
        # prep x and C for the plugging into the model
        x = torch.tensor(x).type(torch.FloatTensor).to(device)
        x = x.view(1, I, T, P)
        C2 = torch.tensor(C).type(torch.FloatTensor).view(1, I, T).to(device)
        
        # plug x and C2 into the model
        with torch.no_grad():
            out = self.forward(x, C2).view(I, T, P).cpu().numpy()
            out = out.transpose(2, 0, 1) # shape (P, I, T)
            probs = np.exp(out) / np.exp(out).sum(axis=0) # shape (P, I, T)
            cum_probs = np.cumsum(probs, axis=0) # shape (P, I, T)
            u = np.random.rand(I, T) # shape (I, T)
            return np.argmax(cum_probs > u, axis=0)     


model = Net().to(device)


def check_list(lst):
    for i in range(len(lst)):
        if lst[i] > 88 or lst[i] < 30:
            lst[i] = lst[i-1]
    return lst


def format_melody(melody):
    
    if len(melody) > 32:
        melody = melody[:32]  # Truncate the list if it's too long
    elif len(melody) < 32:
        melody = melody + [0] * (32 - len(melody))  # Pad the list with zeros if it's too short

    check_list(melody)

    return melody


def read_midi_co(input_dir):
    
    """
        Arg: 
            input_dir: path to midi file to read
        Return:
            melody: list of pitch values in the midi file
    """ 

    midi_stream = converter.parse(input_dir)
    melody = []
    for note in midi_stream.flat.notes:
        pitch = note.pitch.midi
        dur = note.duration.quarterLength
        if dur == 0.25:
            melody.extend([pitch] * 1)
        elif dur == 0.5:
            melody.extend([pitch] * 2)
        elif dur == 0.75:
            melody.append(pitch * 3)
        elif dur == 1.0:
            melody.extend([pitch] * 4)
        elif dur == 1.25:
            melody.extend([pitch] * 5)   
        elif dur == 1.5:
            melody.extend([pitch] * 6)
        elif dur == 1.75:
            melody.extend([pitch] * 7)
        elif dur == 2:
            melody.extend([pitch] * 8)
        elif dur == 2.25:
            melody.extend([pitch] * 9)
        elif dur == 2.5:
            melody.extend([pitch] * 10)
        elif dur == 2.75:
            melody.extend([pitch] * 11)
        elif dur == 3:
            melody.extend([pitch] * 12)
        elif dur == 3.25:
            melody.extend([pitch] * 13)
        elif dur == 3.5:
            melody.extend([pitch] * 14)
        elif dur == 3.75:
            melody.extend([pitch] * 15)  
        elif dur == 4.0:
            melody.extend([pitch] * 16)        
        
    melody = format_melody(melody)
    
    return melody


def get_avg_pitch(melody = []):
    
    if len(melody) == 0:
        return 0
    else:
        return round(sum(melody) / len(melody))
    

def harmonize_soprano(melody_dir, filename, model):
    
    melody = read_midi_co(melody_dir)
    #print(melody)
    y = np.random.randint(P, size=(I, T))
    y[0] = np.array(melody)-30
    D0 = np.ones((1, T)).astype(int)
    D1 = np.zeros((3, T)).astype(int)
    D = np.concatenate([D0, D1], axis=0)

    chorale = Chorale(harmonize(y, D, model))
    #chorale.to_image()
    #plt.show()
    chorale.save(filename, melody_dir)


def harmonize_alto(melody_dir, filename, model):
    
    melody = read_midi_co(melody_dir)
    #print(melody)
    y = np.random.randint(P, size=(I, T))
    y[1] = np.array(melody)-30
    D0 = np.zeros((1, T)).astype(int)
    D1 = np.ones((1, T)).astype(int)
    D2 = np.zeros((2, T)).astype(int)
    D1 = np.concatenate([D0, D1], axis=0)
    D = np.concatenate([D1, D2], axis=0)

    chorale = Chorale(harmonize(y, D, model))
    #chorale.to_image()
    #plt.show()
    chorale.save(filename, melody_dir)


def harmonize_tenor(melody_dir, filename, model):
    
    melody = read_midi_co(melody_dir)
    #print(melody)
    y = np.random.randint(P, size=(I, T))
    y[2] = np.array(melody)-30
    D0 = np.zeros((2, T)).astype(int)
    D1 = np.ones((1, T)).astype(int)
    D2 = np.zeros((1, T)).astype(int)
    D1 = np.concatenate([D0, D1], axis=0)
    D = np.concatenate([D1, D2], axis=0)

    chorale = Chorale(harmonize(y, D, model))
    #chorale.to_image()
    #plt.show()
    chorale.save(filename, melody_dir)


def harmonize_bass(melody_dir, filename, model):
    
    melody = read_midi_co(melody_dir)
    #print(melody)
    y = np.random.randint(P, size=(I, T))
    y[3] = np.array(melody)-30
    D0 = np.zeros((3, T)).astype(int)
    D1 = np.ones((1, T)).astype(int)
    D = np.concatenate([D0, D1], axis=0)

    chorale = Chorale(harmonize(y, D, model))
    #chorale.to_image()
    #plt.show()
    chorale.save(filename, melody_dir)


def harmonize_melody(melody_dir, filename, model):
    
    melody = read_midi_co(melody_dir)
    avg_pitch = get_avg_pitch(melody)
    #print(avg_pitch)

    if avg_pitch >= 30 and avg_pitch <= 47:
        harmonize_bass(melody_dir, filename, model)
    elif avg_pitch >= 48 and avg_pitch <= 61:
        harmonize_tenor(melody_dir, filename, model)
    elif avg_pitch >= 61 and avg_pitch <= 65:
        harmonize_alto(melody_dir, filename, model)
    elif avg_pitch >= 66 and avg_pitch <= 86:
        harmonize_soprano(melody_dir, filename, model)  


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
        #for i in range(-5, 7):  
            #new_data = data.copy()  
            #for j in range(len(new_data)):
                #new_data[j][0] += i
            #population.append(new_data)  
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
    
    return best_individual


def write_midi(arr_list, output_dir):

    """
        Args: 
            arr_list: list of notes [pitch, dur]
            output_dir: output path for midi file
    """ 
    total_dur = 0

    arr = np.array(arr_list).reshape(len(arr_list),2)
    
    notes = [int(note[0]) for note in arr]
    durations = [float(note[1]) for note in arr]

    mstream = stream.Stream()

    for i, nt in enumerate(notes):   
        duration = durations[i]
        pitch_obj = pt.Pitch(nt)
        n = no.Note()
        n.pitch = pitch_obj
        if duration > 0.25: 
            n.duration = du.Duration(duration/2)
        
        ks = key.Key('C')
        mstream.insert(0, ks)
        mm = tempo.MetronomeMark(number=40)
        mstream.insert(0, mm)

        mstream.append(n)

        total_dur += n.duration.quarterLength
        
        if total_dur > 7.5:
            break      
    
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


def read_quant_input(input_dir, output_dir):

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
        
        if len(note_dur_array) > 0 and note_dur_array[-1][0] == pitch:
            note_dur_array[-1][1] += dur
        else:
            note_dur_array.append([pitch, dur])

    note_dur_array = np.array(note_dur_array)
    
    current_sum = np.sum(note_dur_array[:, 1])
    scale_factor = 16 / current_sum
    note_dur_array[:, 1] *= scale_factor
    durs = note_dur_array[:, 1]
    note_dur_array[:, 1] = np.select(
        [durs < 0.375, durs < 0.875, durs < 1.25, durs < 1.75, durs < 2.5, durs < 3.5, True],
        [0.25, 0.5, 1, 1.5, 2, 3, 4]
    )

    note_dur_array = check_total_dur(note_dur_array)

     # Define the notes and durations from the array
    notes = [int(note[0]) for note in note_dur_array]
    durations = [float(note[1]) for note in note_dur_array]

    # Create a Music21 stream to hold the notes
    mstream = stream.Stream()

    # Loop over the notes and add them to the stream
    for i, nt in enumerate(notes):   
        duration = durations[i]
        # Create a Music21 pitch object with the given MIDI number
        pitch_obj = pt.Pitch(nt)
        # Create a Music21 note object with the given pitch and duration
        n = no.Note()
        n.pitch = pitch_obj
        n.duration = du.Duration(duration)
        
        # Set key and tempo 
        ks = key.Key('C')
        mstream.insert(0, ks)
        mm = tempo.MetronomeMark(number=80)
        mstream.insert(0, mm)

        # Add the note to the stream
        mstream.append(n)

    quarterLengthDivisors = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    mstream = mstream.quantize(quarterLengthDivisors)
    
    # Write the stream to a MIDI file
    mstream.write('midi', fp=output_dir)


def adjust_midi_pitch_ranges(filename):
    
    # Load MIDI file
    midi_data = converter.parse(filename)

    # Define pitch ranges for each track
    pitch_ranges = {
        0: [84, 95],   # First track pitch range
        1: [72, 83],   # Second track pitch range
        2: [60, 71],   # Third track pitch range
        3: [48, 59],   # Fourth track pitch range
    }

    # Loop over each track
    for i, track in enumerate(midi_data.parts):
        # Loop over each note in the track
        for note in track.flat.notes:
            # Check if note is within the pitch range
            if note.pitch.midi < pitch_ranges[i][0]:
                # Move note up by an octave until it is in the range
                while note.pitch.midi < pitch_ranges[i][0]:
                    note.pitch.octave += 1
            elif note.pitch.midi > pitch_ranges[i][1]:
                # Move note down by an octave until it is in the range
                while note.pitch.midi > pitch_ranges[i][1]:
                    note.pitch.octave -= 1

    # Save adjusted MIDI file
    midi_data.write("midi", filename)


def run(address, *args):
    
    print("start!")
    input_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/input.mid'
    output_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/output.mid'
    read_quant_input(input_dir, output_dir)
    note_dur_array = read_midi(output_dir)
    input_list = note_dur_array.tolist()

    pop_dir = '/Users/annie/Documents/Shimon-Counterpoint/Genetic_Bach/midiCmaj/4_4/DATA'
    gene_input_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/gene_input.mid'
    gene_output_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/gene_output.mid'

    write_midi(input_list, gene_input_dir)
    best_individual = run_genetic(5, 0.25, 0.02, note_dur_array, pop_dir)
    write_midi(best_individual, gene_output_dir)
    print("finish continuation")
    client.send_message("/playmidi", 1)
    
    melody1_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/gene_input.mid'
    melody2_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/gene_output.mid' 
    filename1 = 'chorale1.mid'
    filename2 = 'chorale2.mid'

    harmonize_melody(melody1_dir, filename1, model)
    harmonize_melody(melody2_dir, filename2, model)

    chorale1_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/chorale1.mid'
    chorale2_dir = '/Users/annie/Documents/Shimon-Counterpoint/midi_files/chorale2.mid'

    adjust_midi_pitch_ranges(chorale1_dir)
    adjust_midi_pitch_ranges(chorale2_dir)
    print("finish harmonization")



if __name__ == '__main__':

    model.load_state_dict(torch.load(base_dir + 'CocoNetModel.pt', map_location=torch.device('mps')))
    
    dispatcher = Dispatcher()
    dispatcher.map("/runGenetic", run)
    
    server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.portServer), dispatcher)
    print("Serving on {}".format(server.server_address))

    server.serve_forever()