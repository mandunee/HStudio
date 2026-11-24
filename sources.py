from contextlib import contextmanager, nullcontext
from functools import partial
import os
from pathlib import Path, PureWindowsPath

from beatmap import MapAssets, Pack


class FolderMaps(MapAssets):
    def __init__(self, root, directory, difficulties):
        self.root = root
        self.directory = directory
        self.difficulties = difficulties
        self.entries = {}
        self.resolved = {}

    def maps(self):
        return self.difficulties

    def read(self, name):
        path = (self.directory / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"File outside the selected folder: {name}")
        return path.read_bytes()

    def find_asset(self, mapname, relative):
        relative = relative.strip('"').replace("\\", "/")
        if relative in self.resolved:
            return self.resolved[relative]
        path = self.directory
        if PureWindowsPath(relative).anchor:
            raise ValueError(f"Expected a relative asset path: {relative}")
        for part in Path(relative).parts:
            if part == "..":
                path = path.parent
            else:
                if path not in self.entries:
                    self.entries[path] = {p.name.casefold(): p for p in path.iterdir()} if path.is_dir() else {}
                path = self.entries[path].get(part.casefold())
                if path is None:
                    break
            path = path.resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"File outside the selected folder: {relative}")
        result = path if path and path.is_file() else None
        self.resolved[relative] = result
        return result


def training_sources(inputs, log):
    roots = sorted({Path(item).resolve() for item in inputs}, key=lambda p: (len(p.parts), str(p)))
    seen = set()
    archives = []
    for root in roots:
        if root in seen:
            continue
        if not root.is_dir():
            if root.suffix.casefold() != ".osz" or not root.is_file():
                raise ValueError(f"Training folder or OSZ file not found: {root}")
            archives.append(root)
            seen.add(root)
            continue
        log(f"Scanning {root}")
        for current, dirs, files in os.walk(root, onerror=lambda err: log(f"Directory scan failed: {err}")):
            directory = Path(current)
            if directory in seen:
                dirs.clear()
                continue
            seen.add(directory)
            dirs[:] = sorted(d for d in dirs if d != "__MACOSX")
            maps = []
            for name in sorted(files):
                suffix = Path(name).suffix.casefold()
                if suffix == ".osu":
                    maps.append(name)
                elif suffix == ".osz":
                    archive = (directory / name).resolve()
                    if archive.is_relative_to(root) and archive not in seen:
                        archives.append(archive)
                        seen.add(archive)
            if maps:
                yield directory, partial(open_folder, root, directory, maps)
    for archive in archives:
        yield archive, partial(open_archive, archive)


def open_folder(root, directory, maps):
    return nullcontext(FolderMaps(root, directory, maps))


@contextmanager
def open_archive(archive):
    with Pack(archive) as pack:
        yield pack
