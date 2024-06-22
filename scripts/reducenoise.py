#!/user/bin/python

import sys
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr
import numpy as np
import time

tstart = time.time()

rate, data = wavfile.read(sys.argv[1])

# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)

# design and apply highpass filter
cutoff = 2000.0
b, a = signal.butter(N=6, Wn=cutoff, btype='highpass', fs=rate)
filtered = signal.filtfilt(b, a, reduced_noise)

# normalize for 16 bit integer output
normalization = int(32000 / max(filtered))

output = (filtered * normalization).astype(np.int16)

wavfile.write(sys.argv[2], rate, output)

elapsed = time.time() - tstart
print(f"Perform noise reduction, filter, normalize in {elapsed:.1f} seconds.")

