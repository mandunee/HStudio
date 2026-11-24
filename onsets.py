from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from scipy.signal import find_peaks, correlate
from scipy.ndimage import median_filter
from audio import RATE, HOP, decode


def analyse(song):
    n = 1024
    padded = np.pad(song,(n//2,n//2))
    frames = np.lib.stride_tricks.sliding_window_view(padded,n)[::HOP]
    window = np.hanning(n)
    blocks = []
    previous = np.zeros(n//2+1)
    for i in range(0,len(frames),512):
        frames_fft = np.log1p(np.abs(np.fft.rfft(frames[i:i+512]*window,axis=1)))
        blocks.append(np.maximum(0,np.diff(frames_fft,axis=0,prepend=previous[None,:])).mean(axis=1))
        previous = frames_fft[-1]
    flux = np.concatenate(blocks)
    novelty = np.maximum(0,flux - median_filter(flux,size=23))
    if novelty.max() < .0001:
        raise ValueError("No clear attacks found. Select a beatmap instead.")
    threshold = max(.002,np.percentile(novelty,85)*.40,novelty.max()*.025)
    peaks,_ = find_peaks(np.pad(novelty,(1,1)),height=threshold,distance=10,prominence=threshold*.45)
    peaks = peaks-1
    times = np.maximum(0,np.rint(peaks*HOP/RATE*1000)).astype(int)
    if len(times)<2:
        raise ValueError("Too few attacks found. Select a beatmap instead.")
    lo,hi = round(60/190*RATE/HOP),round(60/65*RATE/HOP)
    correlation = correlate(novelty,novelty,mode="full",method="fft")[len(novelty)-1:]
    end = min(len(correlation),hi+1)
    lag = lo+np.argmax(correlation[lo:end]) if end>lo else 50
    beat = float(lag)*HOP/RATE*1000
    return times,beat


def draft(input_file, archive):
    path = Path(input_file)
    if path.suffix.lower() not in (".wav",".mp3",".ogg"):
        raise ValueError("Choose an OSZ, WAV, MP3 or OGG file.")
    data = path.read_bytes()
    times,beat = analyse(decode(data))
    anchor = times[0]
    step = beat/4
    times = np.unique(np.rint(anchor + np.rint((times-anchor)/step)*step).astype(int))
    audio = "audio"+path.suffix.lower()
    title = " ".join(path.stem.splitlines())
    lines = ["osu file format v14","", "[General]",f"AudioFilename: {audio}","Mode: 0",
             "", "[Metadata]",f"Title:{title}","Artist:Unknown","Creator:HStudio",
             "Version:Onset Draft","BeatmapID:0","BeatmapSetID:-1","", "[Difficulty]",
             "HPDrainRate:4","CircleSize:4","OverallDifficulty:5","ApproachRate:7",
             "SliderMultiplier:1.4","SliderTickRate:1","", "[TimingPoints]",
             f"{times[0]},{beat:.6f},4,1,0,80,1,0","", "[HitObjects]"]
    xy = [(192,144),(288,176),(320,240),(224,208)]
    for i,t in enumerate(times):
        x,y = xy[i%len(xy)]
        lines.append(f"{x},{y},{t},1,0,0:0:0:0:")
    with ZipFile(archive,"w",compression=ZIP_DEFLATED) as out:
        out.writestr(audio,data)
        out.writestr("Onset Draft.osu","\r\n".join(lines)+"\r\n")
