# Parameters

```py
# Parameters
sample_rate = 44100  # Hz
channels = 1         # mono
duration = 10        # seconds
cry_threshold_dB = 60  # Loudness threshold for detecting suspicious sound
low_freq = 500       # Hz (lower frequency limit for band-pass filter)
high_freq = 4000     # Hz (upper frequency limit for band-pass filter)
use_bandpass_filter = False  # Boolean to enable or disable band-pass filter

# Buffer length in samples
buffer_length = int(sample_rate * 0.5)  # 0.5 seconds buffer

api_url = "http://localhost:5000/audio"  # Replace with your API endpoint
```