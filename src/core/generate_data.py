
import numpy as np

def generate_data(num_samples : int) -> np.ndarray:

    data_array= np.random.randint(0, 100000, size=(num_samples, 2))

    return data_array