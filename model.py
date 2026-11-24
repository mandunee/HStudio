from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED
from pathlib import Path
from tempfile import TemporaryFile
import mmap
import json
import re
import struct
import zlib
import numpy as np
from scipy.spatial import cKDTree
from audio import decode, features, wav, ROLES, KINDS
from beatmap import Map, SETS
from sources import training_sources


def roles(mask):
    return [r for i,r in enumerate(ROLES) if i == 0 or mask & (1 << i)]


def read_audio(pack, path):
    try:
        return decode(pack.read(path))
    except (ValueError, OSError, EOFError, struct.error, BadZipFile, zlib.error) as exc:
        raise ValueError(f"Audio file {path}: {exc}") from exc


class Model:
    def __init__(self, x, y, labels, assets):
        self.x, self.y, self.labels, self.assets = x, y, labels, assets
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0),.30)
        z = (x-self.mean)/self.scale
        self.trees = {}
        for j, kind in enumerate(KINDS):
            ids = np.flatnonzero(x[:, -len(KINDS)+j] > .5)
            if len(ids):
                self.trees[kind] = (cKDTree(z[ids]), ids)
        self.all_tree = (cKDTree(z), np.arange(len(x)))

    def predict(self, x, events=None, workers=1):
        z = (x-self.mean)/self.scale
        output = [None]*len(x)
        for j, kind in enumerate(KINDS):
            query = np.flatnonzero(x[:, -len(KINDS)+j] > .5)
            if len(query) == 0:
                continue
            tree, ids = self.trees.get(kind, self.all_tree)
            k = min(9,len(ids))
            dist, ix = tree.query(z[query], k=list(range(1,k+1)), workers=workers)
            for i, d, near in zip(query, dist, ix):
                votes = Counter()
                for v, j in zip(d,near):
                    votes[int(self.y[ids[j]])] += 1/(.15+v)**2
                lab = self.labels[votes.most_common(1)[0][0]]
                output[i] = lab
        return output

    def save(self, path):
        self.save_examples(path, self.x, self.y, self.labels, self.assets)

    @staticmethod
    def save_examples(path, x, y, labels, assets):
        with ZipFile(path,"x",compression=ZIP_DEFLATED) as out:
            with out.open("examples.npz", "w", force_zip64=True) as arrays:
                np.savez(arrays, x=x, y=y)
            out.writestr("model.json",json.dumps({"format":2,"labels":labels}))
            for key, data in assets.items():
                out.writestr("samples/"+key+".wav",data)

    @classmethod
    def load(cls, path):
        with ZipFile(path) as src:
            meta = json.loads(src.read("model.json"))
            if meta["format"] not in (1, 2):
                raise ValueError("Unsupported model format.")
            with np.load(BytesIO(src.read("examples.npz")),allow_pickle=False) as arrays:
                x, y = arrays["x"],arrays["y"]
            labels = [tuple(v[:4]) for v in meta["labels"]]
            if meta["format"] == 1:
                # Ignore legacy custom-only examples, then merge equivalent stock labels.
                keep = np.array([not lab[4] for lab in meta["labels"]])[y]
                if not keep.all():
                    x, y = x[keep], y[keep]
                if not len(y):
                    raise ValueError("No stock hitsound examples in this model. Train a new model.")
                unique = list(dict.fromkeys(labels))
                lookup = {lab:i for i,lab in enumerate(unique)}
                y = np.asarray([lookup[lab] for lab in labels],dtype=np.int32)[y]
                labels = unique
            return cls(x,y,labels,{})


LEVELS = ("beginner", "easy", "normal", "hard", "insane", "expert", "extra", "ultra")
LEVEL_WORD = re.compile(r"(?<![^\W_])(" + "|".join(LEVELS) + r")(?![^\W_])", re.IGNORECASE)
INVALID_MAP = (OSError, ValueError, IndexError, KeyError, ArithmeticError, BadZipFile,
               EOFError, struct.error, zlib.error)
INVALID_SOURCE = (OSError, BadZipFile, EOFError, struct.error, zlib.error)


def difficulty_rank(version):
    title = re.sub(r"^.*?['’]s\s+", "", version, flags=re.IGNORECASE)
    title = re.split(r"\s+by\s+", title, maxsplit=1, flags=re.IGNORECASE)[0]
    matches = [LEVELS.index(word.lower()) for word in LEVEL_WORD.findall(title)]
    return max(matches) if matches else (len(LEVELS) if version.strip() else -1)


def plan_training(inputs, log):
    plan = []
    for source, open_pack in training_sources(inputs, log):
        best = {}
        try:
            with open_pack() as pack:
                for name in sorted(pack.maps(), key=lambda n: (n.casefold(), n)):
                    try:
                        m = Map(pack.read(name))
                        if not m.standard or m.version != 14:
                            raise ValueError("Requires osu!standard v14")
                        if not m.objects:
                            raise ValueError("No hit objects")
                        song = pack.asset(name, m.keys["AudioFilename"])
                        key = PurePosixPath(name).parent, song
                        version = m.keys.get("Version", "")
                        score = difficulty_rank(version), len(m.objects)
                        if key not in best or score > best[key][0]:
                            best[key] = score, name, version
                    except INVALID_MAP as exc:
                        log(f"Skipping candidate: {source} / {name} · {exc}")
        except INVALID_SOURCE as exc:
            log(f"Skipping file: {source} · {exc}")
            continue
        if best:
            plan.append((source, open_pack, [(name, version) for _, name, version in best.values()]))
    return plan


def describe_error(exc):
    detail = str(exc).strip()
    if not detail and isinstance(exc, MemoryError):
        detail = "Insufficient memory."
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def fit(packs, out, log=print, validate=True, publish=None):
    with TemporaryFile(dir=Path(out).parent) as features_file:
        _fit(packs, out, log, validate, publish, features_file)


def _fit(packs, out, log, validate, publish, features_file):
    groups, id_chunks = [], []
    labels, lookup, assets = [], {}, {}
    count = 0
    skipped = 0
    unpacked_maps = set()
    plan = plan_training(packs, log)
    total = sum(len(selected) for _, _, selected in plan)
    done = 0
    log(f"Scan complete. Selected {total} top difficulties.")
    if total:
        log(f"Processing maps 0.0% (0/{total})")
    for source, open_pack, selected in plan:
        before = done
        try:
            with open_pack() as pack:
                unpacked = source.is_dir()
                audio_cache = {}
                for name, version in selected:
                    location = f"{source} / {name}"
                    try:
                        payload = pack.read(name)
                        m = Map(payload)
                        if not m.standard or m.version != 14:
                            raise ValueError("Requires osu!standard v14")
                        ev = [e for e in m.events() if not e.filename and m.snapped(e)]
                        if not ev:
                            raise ValueError("No on-grid stock hitsound examples")
                        song = pack.asset(name, m.keys["AudioFilename"])
                        if song not in audio_cache:
                            audio = read_audio(pack, song)
                            audio_cache[song] = audio, sha256(wav(audio)).hexdigest()
                        audio, group = audio_cache[song]
                        map_labels = [(e.mask, e.normal, e.addition if e.mask else e.normal, e.volume) for e in ev]
                        if unpacked or unpacked_maps:
                            label_hash = sha256(json.dumps(map_labels).encode()).digest()
                            signature = (sha256(payload).digest(), group, label_hash)
                            if not unpacked and signature in unpacked_maps:
                                skipped += 1
                                log(f"Skipping duplicate archive entry: {location}")
                                continue
                        chunk = features(audio, ev)
                    except INVALID_MAP as exc:
                        skipped += 1
                        log(f"Skipping difficulty: {location} · {exc}")
                        continue
                    finally:
                        done += 1
                        log(f"Processing maps {100*done/total:.1f}% ({done}/{total}) · {version} · {name}")
                    if unpacked:
                        unpacked_maps.add(signature)
                    chunk.tofile(features_file)
                    row_ids = []
                    for label in map_labels:
                        if label not in lookup:
                            lookup[label] = len(labels)
                            labels.append(label)
                        row_ids.append(lookup[label])
                    id_chunks.append(np.asarray(row_ids, dtype=np.int32))
                    groups.append((group, len(ev)))
                    count += 1
                    log(f"Training map: {location} · {len(ev)} events")
        except INVALID_SOURCE as exc:
            skipped += len(selected) - (done - before)
            done = before + len(selected)
            log(f"Skipping file: {source} · {exc}")
            log(f"Processing maps {100*done/total:.1f}% ({done}/{total})")
    if not id_chunks:
        raise ValueError("No usable osu!standard v14 maps found. Check your maps and audio files.")
    if skipped:
        log(f"Skipped difficulties: {skipped}")
    log("Map processing complete. Saving model.")
    y = np.concatenate(id_chunks)
    del id_chunks
    songs = sorted({group for group, _ in groups})
    features_file.flush()
    with mmap.mmap(features_file.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
        x = np.ndarray((len(y), chunk.shape[1]), dtype=np.float32, buffer=mapping)
        Model.save_examples(out, x, y, labels, assets)
        if publish:
            publish()
        log(f"Model saved: {count} difficulties, {len(songs)} songs, {len(y)} events")
        if not validate:
            log("Evaluation skipped.")
        else:
            try:
                evaluate(x, y, labels, groups, songs, log)
            except Exception as exc:
                log(f"Evaluation failed: {describe_error(exc)}. The saved model is available.")


def evaluate(x, y, labels, groups, songs, log):
    if len(songs) > 1:
        hold = set(songs[:max(1,round(len(songs)*.2))])
        test = np.repeat([group in hold for group, _ in groups], [size for _, size in groups])
        train = ~test
        rows = np.flatnonzero(test)
        log(f"Building evaluation index for {train.sum():,} training examples")
        small = Model(x[train],y[train],labels,{})
        pred = np.empty(len(rows), dtype=np.uint8)
        log(f"Evaluating 0.0% (0/{len(rows):,})")
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            guessed = small.predict(x[batch], workers=8)
            end = start + len(batch)
            pred[start:end] = [label[0] for label in guessed]
            log(f"Evaluating {100*end/len(rows):.1f}% ({end:,}/{len(rows):,})")
        del small
        actual = np.array([labels[i][0] for i in y[test]])
        mode = Counter(labels[i][0] for i in y[train]).most_common(1)[0][0]
        log(f"Held-out evaluation: {len(hold)} songs, {len(actual)} events, exact match "
            f"{np.mean(actual==pred):.1%} · majority baseline {np.mean(actual==mode):.1%}")
        for bit,role in ((2,"Whistle"),(4,"Finish"),(8,"Clap")):
            a, b = (actual & bit)>0,(pred & bit)>0
            denom = int(a.sum()+b.sum())
            if denom:
                log(f"  {role} F1 {2*np.count_nonzero(a & b)/denom:.3f} · support {a.sum()}")
    else:
        log("Evaluation skipped: only one distinct song.")
