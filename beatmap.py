from bisect import bisect_right
from dataclasses import dataclass
from pathlib import PurePosixPath
from zipfile import ZipFile
import posixpath
import re

SETS = ("", "normal", "soft", "drum")


def number(value, default):
    return float(value) if value != "" else default


def fields(raw, count, separator=":"):
    parts = raw.split(separator, count - 1)
    return parts + [""] * (count - len(parts))


@dataclass(frozen=True)
class Point:
    time: float
    beat: float
    anchor: float
    meter: int
    velocity: float
    bank: int
    volume: int
    index: int


@dataclass(frozen=True)
class Event:
    line: int
    node: int
    kind: str
    time: float
    point: Point
    mask: int
    normal: int
    addition: int
    volume: int
    index: int
    filename: str


class Map:
    def __init__(self, payload):
        self.lines = payload.decode("utf-8-sig").splitlines()
        if not self.lines or not re.fullmatch(r"osu file format v\d+", self.lines[0].strip()):
            raise ValueError("Invalid osu! file header.")
        self.version = int(self.lines[0].strip().split("v")[-1])
        self.keys, self.key_lines, self.objects = {}, {}, {}
        timing = []
        self.timing_rows = []
        self.sample_events = set()
        self.first_red = None
        section = ""
        for i, raw in enumerate(self.lines):
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
            elif not line or line.startswith("//"):
                continue
            elif section == "TimingPoints":
                timing.append(line.split(","))
                self.timing_rows.append(i)
            elif section == "Events" and line.split(",", 1)[0].strip().lower() in ("sample", "5"):
                self.sample_events.add(i)
            elif section == "HitObjects":
                self.objects[i] = line.split(",")
            elif section in ("General", "Metadata", "Difficulty", "Editor") and ":" in line:
                key, value = line.split(":", 1)
                self.keys[key.strip()] = value.strip()
                self.key_lines[key.strip()] = i
        self.standard = self.keys.get("Mode", "0") == "0"
        base = self.keys.get("SampleSet", "Normal").lower()
        bank = SETS.index(base) if base in SETS[1:] else 1
        self.points = [Point(float("-inf"), 500, 0, 4, 1, bank, 100, 0)]
        for row in sorted(timing, key=lambda x: float(x[0])):
            row = row + [""] * 8
            t, length = map(float, row[:2])
            prev = self.points[-1]
            red = number(row[6], 1) == 1
            if (red and length <= 0) or (not red and length >= 0):
                continue
            if red and self.first_red is None:
                self.first_red = t
            self.points.append(Point(t, length if red else prev.beat, t if red else prev.anchor,
                                     max(1, int(number(row[2], 4))) if red else prev.meter,
                                     1 if red else min(10, max(.1, -100 / length)),
                                     int(number(row[3], 0)) or bank, int(number(row[5], 100)), int(number(row[4], 0))))
        self.divisor = int(self.keys.get("BeatDivisor", "4"))
        if self.divisor <= 0:
            self.divisor = 4
        self.times = [p.time for p in self.points]

    def point(self, t):
        return self.points[bisect_right(self.times, t) - 1]

    def snapped(self, event):
        if self.first_red is None or event.time < self.first_red:
            return False
        step = event.point.beat / self.divisor
        phase = (event.time-event.point.anchor) / step
        return abs(phase-round(phase))*step <= 1.0 + 1e-8

    def events(self):
        events = []
        for line, row in self.objects.items():
            t, flags, mask = float(row[2]), int(row[3]), int(row[4]) & 14
            nodes = []
            if flags & 1:
                slot = 5
                nodes.append((t, "circle", mask, ""))
            elif flags & 2:
                slot = 10
                count, length = int(row[6]), float(row[7])
                if not 1 <= count <= 10000:
                    raise ValueError("Invalid slider repeat count.")
                point = self.point(t)
                mult = float(self.keys.get("SliderMultiplier", "1.4"))
                span = length * point.beat / (100 * mult * point.velocity)
                masks = row[8].split("|") if len(row) > 8 else []
                sets = row[9].split("|") if len(row) > 9 else []
                for j in range(count + 1):
                    kind = "head" if j == 0 else ("tail" if j == count else "repeat")
                    sound = int(masks[j]) if j < len(masks) and masks[j] else mask
                    nodes.append((t + j * span, kind, sound & 14, sets[j] if j < len(sets) else ""))
            elif flags & 8:
                slot = 6
                nodes.append((float(row[5]), "spinner", mask, ""))
            else:
                continue
            sample = fields(",".join(row[slot:]), 5) if len(row) > slot else [""] * 5
            for j, (time, kind, sound, edge) in enumerate(nodes):
                p = self.point(time)
                sound_point = self.point(time + 5)
                slider = bool(flags & 2)
                pair = fields(edge, 2) if slider and edge else sample[:2]
                normal = int(pair[0] or 0) or sound_point.bank
                addition = int(pair[1] or 0) or normal
                vol = sound_point.volume if slider else int(sample[3] or 0) or sound_point.volume
                idx = sound_point.index if slider else int(sample[2] or 0) or sound_point.index
                events.append(Event(line, j, kind, time, p, sound, min(3,max(1,normal)),
                                    min(3,max(1,addition)), min(100,max(0,vol)), idx, "" if slider else sample[4]))
        return sorted(events, key=lambda e: (e.time, e.line, e.node))

    def render(self, events, labels, version):
        lines = self.lines.copy()
        for i in self.timing_rows:
            row = lines[i].split(",")
            if len(row) > 4:
                row[4] = "0"
                lines[i] = ",".join(row)
        for i in self.sample_events:
            lines[i] = ""
        groups = {}
        for event, label in zip(events, labels, strict=True):
            groups.setdefault(event.line, []).append((event, label))
        for line, group in groups.items():
            group.sort(key=lambda pair: pair[0].node)
            row = self.objects[line].copy()
            e, label = group[0]
            mask, normal, add, volume = label
            if e.kind == "head":
                row = (row + [""] * 11)[:11]
                row[4] = "0"
                row[10] = f"{normal}:{add}:0:0:"
                row[8] = "|".join(str(lab[0]) for _, lab in group)
                row[9] = "|".join(f"{lab[1]}:{lab[2]}" for _, lab in group)
            else:
                row[4] = str(mask)
                slot = 5 if e.kind == "circle" else 6
                row = (row + [""])[:slot+1]
                row[slot] = f"{normal}:{add}:0:{max(1,volume)}:"
            lines[line] = ",".join(row)
        if "Version" in self.key_lines:
            lines[self.key_lines["Version"]] = "Version:" + version
        else:
            idx = next(i for i,s in enumerate(lines) if s.strip() == "[Metadata]")
            lines.insert(idx+1,"Version:" + version)
        for i,s in enumerate(lines):
            if s.split(":",1)[0].strip() == "BeatmapID":
                lines[i] = "BeatmapID:0"
        return ("\r\n".join(lines) + "\r\n").encode("utf-8")


class MapAssets:
    def asset(self, mapname, relative):
        key = self.find_asset(mapname, relative)
        if key is None:
            raise ValueError(f"Missing beatmap asset: {relative}")
        return key


class Pack(ZipFile, MapAssets):
    def __enter__(self):
        super().__enter__()
        self.names = {n.replace("\\", "/").casefold(): n for n in self.namelist()}
        return self

    def maps(self):
        return sorted(n for n in self.namelist() if n.lower().endswith(".osu")
                      and not n.startswith("__MACOSX/"))

    def find_asset(self, mapname, relative):
        path = posixpath.normpath(posixpath.join(str(PurePosixPath(mapname).parent),
                                                 relative.replace("\\", "/").strip('"')))
        return self.names.get(path.casefold())
