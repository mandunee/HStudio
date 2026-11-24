# HStudio

HStudio learns stock hitsound choices from paired osu!standard v14 beatmaps and their music. It exports a new difficulty in a local `hitsounded.osz` archive. Train a model from the command line, then generate hitsounds in the desktop app or CLI.

## Installation

Use Python 3.10 or later. From the project directory, install the dependencies with:

```sh
python -m pip install -r requirements.txt
```

Install FFmpeg on your PATH to decode MP3 and OGG audio. The desktop app also requires Tkinter (`python3-tk` on Debian and Ubuntu).

## Train a model

```sh
python studio.py train "Songs" -o my-style.hsm
```

Point the command at a parent folder such as `Songs`, individual map folders, or one or more OSZ archives. It scans recursively and resolves each difficulty's audio relative to its map. It chooses one difficulty per song in each map folder. Named difficulties rank in ascending order from Beginner through Easy, Normal, Hard, Insane, Expert, Extra and Ultra. A custom title with no recognized rank takes priority. Guest names are supported. Ties use hit object count, then filename.

For each eligible event, the tool pairs spectral energy and attack changes with beat position, spacing, object type, sample set, volume and hitsound flags. Named custom samples are excluded. The model uses up to nine nearby examples and distance-weighted voting. All training and prediction run locally on the CPU.

Training progress counts processed difficulties, including skipped maps. Features are temporarily written to the output drive. The model is saved before evaluation on held-out songs. Add `--skip-validation` to skip scoring. Choose a new output filename for each run.

## Generate hitsounds

```sh
python app.py
```

1. Select your `.hsm` model.
2. Select a beatmap archive, a single `.osu` file, or an unpacked map folder. Choose the target difficulty from the list.
3. Select **Generate** and choose an output location. The activity log shows the selected difficulty, audio analysis, hitsound assignment, and archive export. **Open output folder** opens the completed result.

You can also select a WAV, MP3 or OGG file. If the same folder contains difficulties referencing that audio, the app lists them. Choose **Create a timing draft from the audio** to generate a simple circle layout on an estimated beat grid.

For command-line generation:

```sh
python studio.py list "input.osz"
python studio.py apply my-style.hsm "input.osz" -d "Hard" -o result
python studio.py apply my-style.hsm "song.mp3" -o song-result
```

Use the internal .osu filename from `list` if difficulty names overlap. The output directory must be new.

## Timing and output

Each red timing point starts a new beat grid. Inherited timing points affect slider velocity and edge times. The selected difficulty's editor BeatDivisor determines the subdivision, with a default of 1/4 beat. Circles, slider edges and spinner ends within 1 ms of the grid may receive whistle, finish or clap additions. Off-grid objects keep their positions and play the base hit. Audio-only drafts use quarter-beat subdivisions.

The archive contains the generated difficulty, its soundtrack and supported visual assets. Hitsounds use the game's stock sample references and reflect the player's skin. The original map is left untouched.

This project is for hobby and research use only. It must not be used on beatmaps intended for upload to the osu! website.
