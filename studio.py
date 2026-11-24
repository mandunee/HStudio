from argparse import ArgumentParser
from collections import Counter
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED
import os
import sys
from audio import decode, features
from beatmap import Map, Pack, MapAssets
from model import describe_error, Model, fit, roles
from onsets import draft


class TargetFolder(MapAssets):
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.names = {}
        for path in self.root.rglob("*"):
            if path.is_file() and path.resolve().is_relative_to(self.root):
                name = path.relative_to(self.root).as_posix()
                self.names[name.casefold()] = name

    def namelist(self):
        return self.names.values()

    def maps(self):
        return sorted(n for n in self.namelist() if "/" not in n and n.lower().endswith(".osu"))

    def read(self, name):
        return (self.root / name).read_bytes()

    find_asset = Pack.find_asset


def open_target(path):
    path = Path(path)
    if path.is_dir() or path.suffix.lower() == ".osu":
        return nullcontext(TargetFolder(path if path.is_dir() else path.parent))
    return Pack(path)


def eligible_maps(pack):
    for name in pack.maps():
        try:
            m = Map(pack.read(name))
        except (ValueError, IndexError, OSError):
            continue
        if m.standard and m.version == 14:
            yield name, m


def choices(path):
    with open_target(path) as pack:
        return [name for name, _ in eligible_maps(pack)]


def song_maps(path):
    path = Path(path)
    if not any(p.suffix.lower() == ".osu" for p in path.parent.iterdir()):
        return []
    pack = TargetFolder(path.parent)
    return [name for name, m in eligible_maps(pack)
            if pack.find_asset(name, m.keys.get("AudioFilename", "")) == path.name]


def choose(pack, difficulty):
    eligible = list(eligible_maps(pack))
    if difficulty:
        eligible = [(n,m) for n,m in eligible if n == difficulty or m.keys.get("Version") == difficulty]
    if len(eligible) != 1:
        raise ValueError("Select one osu!standard v14 difficulty.\n" +
                         "\n".join(n for n,_ in eligible))
    return eligible[0]


def unique_name(names, desired):
    p = PurePosixPath(desired)
    n = str(p)
    i = 1
    while n.casefold() in names:
        i += 1
        n = str(p.with_name(f"{p.stem}-{i}{p.suffix}"))
    names.add(n.casefold())
    return n


def export(model, pack, name, m, out, log):
    log(f"Reading audio for {name}")
    song = decode(pack.read(pack.asset(name,m.keys["AudioFilename"])))
    ev = m.events()
    if not ev:
        raise ValueError("This difficulty contains no hit objects.")
    if m.first_red is None:
        raise ValueError("The selected difficulty has no uninherited timing point.")
    eligible = [i for i,e in enumerate(ev) if m.snapped(e)]
    labels = [(0,e.normal,e.normal,e.volume) for e in ev]
    if eligible:
        log(f"Analyzing {len(ev):,} events")
        x = features(song, ev)
        for start in range(0, len(eligible), 256):
            batch = eligible[start:start+256]
            for i, label in zip(batch, model.predict(x[batch]), strict=True):
                labels[i] = label
            log(f"Assigned hitsounds to {start+len(batch):,}/{len(eligible):,} snapped events")
    log(f"1/{m.divisor} beat grid: {len(eligible)} on-grid, {len(ev)-len(eligible)} off-grid")
    folder = PurePosixPath(name).parent
    used = set(pack.names)
    version = m.keys.get("Version","Normal") + " (AI Hitsounds)"
    known = {level.keys.get("Version","") for _,level in eligible_maps(pack)}
    while version in known:
        version += "+"
    new_map = unique_name(used,folder / (PurePosixPath(name).stem+" (AI Hitsounds).osu"))
    music = pack.asset(name,m.keys["AudioFilename"])
    visual_extensions = {".jpg",".jpeg",".png",".gif",".webp",".mp4",".avi",".flv",".mov",".mkv",".webm",".m4v"}
    log("Writing beatmap archive")
    with Pack(out/"hitsounded.osz","w",compression=ZIP_DEFLATED) as result:
        for info in pack.namelist():
            suffix = PurePosixPath(info).suffix.lower()
            if info == music or suffix in visual_extensions:
                result.writestr(info, pack.read(info))
            elif suffix == ".osb":
                lines = pack.read(info).splitlines(keepends=True)
                result.writestr(info, b"".join(line for line in lines if
                    line.lstrip(b"\xef\xbb\xbf \t").split(b",",1)[0].strip().lower() not in (b"sample",b"5")))
        result.writestr(new_map, m.render(ev,labels,version))
    hist = Counter(role for lab in labels for role in roles(lab[0]))
    log(f"{version} · {len(ev)} events · " + " / ".join(f"{k} {v}" for k,v in hist.items()))


def apply(model_path, pack_path, difficulty, output, log=print):
    output = Path(output)
    if output.exists():
        raise ValueError("The output folder already exists. Choose a new name.")
    output.parent.mkdir(parents=True,exist_ok=True)
    log(f"Loading model {Path(model_path).name}")
    model = Model.load(model_path)
    with TemporaryDirectory(dir=output.parent) as temp:
        target = Path(pack_path)
        song_only = not target.is_dir() and target.suffix.lower() not in (".osz", ".osu")
        if not difficulty and target.is_file() and target.suffix.lower() == ".osu":
            difficulty = target.name
        if song_only:
            source = Path(temp)/"draft.osz"
            draft(pack_path,source)
            log("Created an onset-based timing draft.")
        else:
            source = pack_path
        stage = Path(temp)/"result"
        stage.mkdir()
        with open_target(source) as pack:
            name,m = choose(pack,difficulty)
            export(model,pack,name,m,stage,log)
        stage.rename(output)
    log(f"Exported to {output}")


def train(packs, out, log=print, validate=True):
    out = Path(out)
    if out.exists():
        raise ValueError("A model already exists at this path. Choose a new name.")
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        staged = Path(temp)/"model.hsm"
        fit(packs, staged, log, validate=validate, publish=lambda: os.link(staged,out))


def main():
    parser = ArgumentParser(description="HStudio for osu!standard")
    sub = parser.add_subparsers(dest="cmd",required=True)
    p = sub.add_parser("train",help="Train a model from Songs folders or OSZ archives")
    p.add_argument("--skip-validation", action="store_true", help="Skip evaluation after saving the model")
    p.add_argument("packs",nargs="+",type=Path)
    p.add_argument("-o","--output",required=True,type=Path)
    p = sub.add_parser("list",help="List supported difficulties")
    p.add_argument("pack")
    p = sub.add_parser("apply",help="Export a hitsounded difficulty")
    p.add_argument("model")
    p.add_argument("pack")
    p.add_argument("-d","--difficulty")
    p.add_argument("-o","--output",required=True)
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.packs,args.output,log=lambda message: print(message, flush=True), validate=not args.skip_validation)
    elif args.cmd == "list":
        print("\n".join(choices(args.pack)))
    else:
        apply(args.model,args.pack,args.difficulty,args.output)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        sys.exit(f"Error: {describe_error(exc)}")
