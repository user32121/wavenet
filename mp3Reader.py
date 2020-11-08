import pydub 
import numpy as np

def read(f, normalized=False, allow2Channels=True):
    """MP3 to numpy array
    
        Arguments:
            f: filename
            normalized: whether the resulting values are scaled to [-1,1) range
        Returns:
            audio rate,
            array with sound values
    """
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
        if(not allow2Channels):
            y = y.mean(axis=1)
    
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3
    
        Argyments:
            f: filename
            sr: audio rate
            x: array with sound values
            normalizeed: whether the values are scaled to [-1,1) range
        Returns:
            Audiosegment constructed from the array
    """
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
    return song

def readFFT(f, allow2Channels=True):
    """MP3 to fft numpy array
    
        Arguments:
            f: filename
        Returns:
            audio rate,
            array with fft values
    """
    frequency, ar = read(f, allow2Channels=allow2Channels)
    return frequency, np.fft.fft(ar, axis=0)

def writeFFT(f, sr, x):
    """numpy array to MP3
    
        Argyments:
            f: filename
            sr: audio rate
            x: array with sound values as fft
            normalizeed: whether the values are scaled to [-1,1) range
        Returns:
            Audiosegment constructed from the array
    """
    ar = np.fft.ifft(x)
    return write(f,sr,ar)

