import numpy as np

def one_hot_encode(sequences):
    sequence_length = len(sequences[0])
    batch_size = len(sequences)
    encoding = np.zeros((batch_size, sequence_length, 4), dtype=np.float32)
    
    base_to_index = np.zeros(128, dtype=np.int8)
    base_to_index[ord('A')] = 0
    base_to_index[ord('C')] = 1
    base_to_index[ord('G')] = 2
    base_to_index[ord('T')] = 3
    
    for i, seq in enumerate(sequences):
        indices = base_to_index[np.frombuffer(seq.encode(), dtype=np.int8)]
        valid_bases = indices >= 0
        encoding[i, np.arange(len(seq))[valid_bases], indices[valid_bases]] = 1
        
    return encoding