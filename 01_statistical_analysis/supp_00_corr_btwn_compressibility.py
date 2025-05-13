"""
Compute Pearson correlations between JPEG-based and Canny-based compressibility scores
for image sets 1 through 4.

Output:
    Prints the Pearson correlation for each image set to the console.
"""

import os
import sys
sys.path.append(os.path.join('..', 'scripts'))

from models import load_data
from scipy.stats import pearsonr

def main():
    for image_set_index in [1, 2, 3, 4]:
        df = load_data(image_set_index)
        corr, _ = pearsonr(df['jpeg'], df['canny'])
        print(f"Pearson correlation between jpeg and canny for set {image_set_index}: {corr}")

if __name__ == '__main__':
    main()