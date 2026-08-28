import numpy as np
import struct

def load_images(path):
    with open(path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        
        images = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )
        
        images = images.reshape(num_images, rows, cols) 
        
    return images

def load_labels(path):
    with open(path, 'rb') as f:
        magic_num, num_labels = struct.unpack('>II', f.read(8))
    
        labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )
    
    return labels