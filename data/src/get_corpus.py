from music21 import *
import os
from pathlib import Path

# Define paths
#corpus_path = corpus.getComposer('bach')
#print(corpus_path)
corpus_path = '/Users/annie/Desktop/Bach'

# Iterate over the corpus files
for p_bach in os.listdir(corpus_path):
    if p_bach.endswith(".mxl"): # assumes that the corpus files are in MusicXML format
        print("Parsing file:", p_bach)
        print(p_bach)
        bach_chorale = converter.parse(os.path.join(corpus_path, p_bach))

        # Transpose to C major or A minor
        original_key = bach_chorale.analyze('key')
        if original_key.mode == "major":
            key_diff = interval.Interval(original_key.tonic, pitch.Pitch('C'))
        else:
            key_diff = interval.Interval(original_key.tonic, pitch.Pitch('A'))
        bach_chorale_transposed = bach_chorale.transpose(key_diff)

        # Write to MIDI file
        p_bach = Path(p_bach)
        midi_filename = p_bach.stem + "_transposed.mid"
        bach_chorale_transposed.write('midi', midi_filename)
        print("Transposed chorale written to MIDI file:", midi_filename)