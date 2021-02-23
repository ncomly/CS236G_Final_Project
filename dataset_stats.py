from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, isdir, join

path = '.'
directories = [d for d in listdir(path) if isdir(join(path, d))]


print(len(directories))
counts = np.zeros(len(directories))
for i, directory in tqdm(enumerate(directories)):

    directory_path = join(path, directory)
    try:
        files = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    except:
        print(f'failed: {directory}')

    counts[i] = len(files)

print(max(counts))
print(min(counts))
print(sum(counts)/len(counts))
print(np.median(counts))
print(len(np.where(counts >= 5)[0]))

