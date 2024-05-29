# Select x random filenames from a directory and save them in a list

import os
import random

def select_video_subset(path, x):
    filenames = os.listdir(path)
    random.shuffle(filenames)
    return filenames[:x]

# Example usage
