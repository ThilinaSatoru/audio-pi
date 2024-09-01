import numpy as np
import requests
import scipy.signal as signal
import sounddevice as sd
import scipy.io.wavfile as wavfile
import time

# Parameters
sample_rate = 44100  # Hz
channels = 1  # mono
duration = 10  # seconds
cry_threshold_dB = 60  # Loudness threshold for detecting suspicious sound
low_freq = 500  # Hz (lower frequency limit for band-pass filter)
high_freq = 4000  # Hz (upper frequency limit for band-pass filter)
use_bandpass_filter = False  # Boolean to enable or disable band-pass filter

# Buffer length in samples
buffer_length = int(sample_rate * 0.5)  # 0.5 seconds buffer

api_url = "http://localhost:5000/audio"  # Replace with your API endpoint


def bandpass_filter(data, low_freq, high_freq, sample_rate):
    """Apply a band-pass filter to the audio data."""
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def dB_to_amplitude(dB):
    """Convert dB SPL to amplitude level."""
    return 10 ** (dB / 20)


def amplitude_to_dB(amplitude):
    """Convert amplitude level to dB SPL."""
    return 20 * np.log10(amplitude)


def check_loudness(data, threshold_dB):
    """Check if the audio data exceeds the loudness threshold."""
    data = np.array(data)
    max_amplitude = np.max(np.abs(data))
    threshold_amplitude = dB_to_amplitude(threshold_dB)
    return max_amplitude > threshold_amplitude


def record_audio(file_name, duration, sample_rate, channels):
    """Record audio and save to file."""
    print(f"Recording {file_name}...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()

    # Save the recording to a WAV file using scipy
    wavfile.write(file_name, sample_rate, audio)
    print(f"Recording complete: {file_name}")


def send_audio_file(file_path, api_url):
    """Send the recorded WAV file to a POST API."""
    with open(file_path, 'rb') as f:
        files = {'file': ('test.wav', f, 'audio/wav')}
        response = requests.post(api_url, files=files)
    print(f"Response: {response.status_code}, {response.text}")


def monitor_audio():
    """Monitor audio for suspicious sounds and start recording if detected."""
    print("Starting audio monitor...")

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16', blocksize=buffer_length) as stream:
        while True:
            data, overflowed = stream.read(buffer_length)
            if overflowed:
                print("Warning: Audio buffer overflow!")

            if use_bandpass_filter:
                # Apply band-pass filter to focus on the relevant frequency range
                filtered_data = bandpass_filter(data.flatten(), low_freq, high_freq, sample_rate)
            else:
                # Use raw data if filtering is disabled
                filtered_data = data.flatten()

            # Calculate and print the dB SPL of the audio
            max_amplitude = np.max(np.abs(filtered_data))
            if max_amplitude > 0:
                dB = amplitude_to_dB(max_amplitude)
                print(f"Current dB SPL: {dB:.2f} dB")

            # Check if loudness exceeds the threshold
            if check_loudness(filtered_data, cry_threshold_dB):
                # Start recording when suspicious sound is detected
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                file_name = f"suspicious_sound_{timestamp}.wav"
                record_audio(file_name, duration, sample_rate, channels)
                print(f"Suspicious sound detected and recorded to {file_name}")
                # Send the recorded file
                send_audio_file(file_name, api_url)

            # Short delay to prevent excessive CPU usage
            time.sleep(0.1)


if __name__ == "__main__":
    try:
        monitor_audio()
    except KeyboardInterrupt:
        print("\nExiting audio monitor...")
        exit(0)
