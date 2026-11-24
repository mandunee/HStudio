from io import BytesIO
from math import gcd
import struct
import shutil
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

RATE = 22050
HOP = 220
ROLES = ("normal", "whistle", "finish", "clap")
KINDS = ("circle", "head", "repeat", "tail", "spinner")


def decode(data):
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        try:
            rate, x = wavfile.read(BytesIO(data))
            kind = x.dtype.kind
            if kind == "u":
                x = (x.astype(np.float32) - 128) / 128
            elif kind == "i":
                x = x.astype(np.float32) / (2 ** (8*x.dtype.itemsize-1))
            else:
                x = x.astype(np.float32)
            if x.ndim == 2:
                x = x.mean(axis=1)
            d = gcd(rate, RATE)
            if rate != RATE:
                x = resample_poly(x, RATE//d, rate//d)
            if len(x) and np.isfinite(x).all():
                return x.astype(np.float32)
        except (ValueError, OSError, EOFError, struct.error):
            pass
    exe = shutil.which("ffmpeg")
    if not exe:
        raise ValueError("Install FFmpeg and add it to PATH to decode this audio.")
    p = subprocess.run([exe, "-v", "error", "-i", "pipe:0", "-vn", "-ac", "1",
                        "-ar", str(RATE), "-f", "f32le", "pipe:1"], input=data, capture_output=True)
    if p.returncode or not p.stdout:
        raise ValueError("Could not decode audio. " + p.stderr.decode(errors="replace")[:200])
    return np.frombuffer(p.stdout, dtype="<f4")


def wav(x):
    buf = BytesIO()
    wavfile.write(buf, RATE, np.rint(np.clip(x, -1, 1) * 32767).astype("<i2"))
    return buf.getvalue()


def features(x, events):
    size = 1024
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, (size//2, size//2)), size)[::HOP]
    cuts = np.searchsorted(np.fft.rfftfreq(size, 1/RATE), [30,90,180,350,700,1400,2800,5600,10000])
    window = np.hanning(size).astype(np.float32)
    chunks = []
    for start in range(0, len(frames), 512):
        power = np.abs(np.fft.rfft(frames[start:start+512] * window, axis=1)) ** 2
        chunks.append(np.stack([np.log1p(power[:,lo:hi].mean(axis=1))
                                for lo,hi in zip(cuts[:-1],cuts[1:])], axis=1))
    bands = np.concatenate(chunks)
    bands -= bands.mean(axis=0)
    times = np.array([e.time for e in events])
    positions = np.rint(times * RATE / (1000*HOP)).astype(int)
    blocks = []
    for left, right in ((-15,-5),(-4,-1),(0,3),(4,10)):
        ids = np.clip(positions[:,None] + np.arange(left,right+1), 0,len(bands)-1)
        blocks.append(bands[ids].mean(axis=1))
    blocks.append(blocks[2] - blocks[1])
    rhythm = []
    for i,e in enumerate(events):
        beat = e.point.beat
        phase = (e.time-e.point.anchor)/beat
        gaps = [(e.time-times[i-1])/beat if i else 1,
                (times[i+1]-e.time)/beat if i+1<len(events) else 1]
        rhythm.append([np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),
                       np.sin(2*np.pi*phase/e.point.meter),np.cos(2*np.pi*phase/e.point.meter),
                       *np.log1p(np.clip(gaps,0,8)), np.log(beat/500),
                       *[float(e.kind==k) for k in KINDS]])
    return np.column_stack([*blocks,rhythm]).astype(np.float32)
