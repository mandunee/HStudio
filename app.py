import os
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from threading import Thread
from tkinter import filedialog, messagebox, ttk

from model import describe_error
from studio import apply, choices, song_maps

DRAFT = "Create a timing draft from the audio"


class Studio:
    def __init__(self, root):
        self.root = root
        root.title("HStudio")
        root.geometry("740x510")
        root.minsize(600, 420)
        self.model = tk.StringVar()
        self.target = tk.StringVar()
        self.difficulty = tk.StringVar(value="Select a beatmap first")
        self.status = tk.StringVar(value="Ready")
        self.messages = queue.Queue()
        self.busy = False
        self.last = None

        frame = ttk.Frame(root, padding=(24, 18))
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(6, weight=1)
        ttk.Label(frame, text="HStudio", font=("TkDefaultFont", 17, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 16))
        self.inputs = []
        for row, label, variable, command in (
            (1, "Model", self.model, self.pick_model),
            (2, "Beatmap or audio", self.target, self.pick_target),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 14))
            ttk.Entry(frame, textvariable=variable, state="readonly").grid(
                row=row, column=1, sticky="ew", pady=4)
            button = ttk.Button(frame, text="Browse…", command=command)
            button.grid(row=row, column=2, padx=(6, 0))
            self.inputs.append(button)
        folder = ttk.Button(frame, text="Map folder…", command=self.pick_folder)
        folder.grid(row=2, column=3, padx=(4, 0))
        self.inputs.append(folder)
        ttk.Label(frame, text="Difficulty").grid(row=3, column=0, sticky="w")
        self.levels = ttk.Combobox(frame, textvariable=self.difficulty, state="disabled")
        self.levels.grid(row=3, column=1, columnspan=3, sticky="ew", pady=4)

        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(12, 18))
        self.start = ttk.Button(actions, text="Generate…", command=self.generate)
        self.start.pack(side="left")
        self.open_button = ttk.Button(actions, text="Open output folder", command=self.open_output)
        self.open_button.pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.status).pack(side="right")
        ttk.Label(frame, text="Activity log").grid(row=5, column=0, columnspan=4, sticky="w", pady=(0, 5))

        panel = ttk.Frame(frame)
        panel.grid(row=6, column=0, columnspan=4, sticky="nsew")
        self.log = tk.Text(panel, wrap="word", state="disabled", height=8, borderwidth=0,
                           background="#202427", foreground="#e4e8e9", padx=12, pady=10,
                           font="TkFixedFont")
        scroll = ttk.Scrollbar(panel, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.refresh()
        root.after(100, self.poll)

    def refresh(self):
        for button in self.inputs:
            button.configure(state="disabled" if self.busy else "normal")
        self.levels.configure(state="readonly" if self.target.get() and not self.busy else "disabled")
        ready = self.model.get() and self.target.get() and not self.busy
        self.start.configure(state="normal" if ready else "disabled")
        self.open_button.configure(state="normal" if self.last and not self.busy else "disabled")

    def pick_model(self):
        path = filedialog.askopenfilename(title="Select model", filetypes=[("Hitsound model", "*.hsm")])
        if path:
            self.model.set(path)
            self.refresh()

    def pick_target(self):
        self.select_target(filedialog.askopenfilename(title="Select beatmap or audio", filetypes=[
            ("Beatmaps and audio", "*.osz *.osu *.mp3 *.ogg *.wav")]))

    def pick_folder(self):
        self.select_target(filedialog.askdirectory(title="Select a folder containing .osu files", mustexist=True))

    def select_target(self, value):
        if not value:
            return
        path = Path(value)
        try:
            is_map = path.is_dir() or path.suffix.lower() in (".osu", ".osz")
            names = choices(path) if is_map else song_maps(path) + [DRAFT]
            selected = path.name if path.is_file() and path.suffix.lower() == ".osu" else next(iter(names), None)
            if selected not in names:
                raise ValueError("Choose a map set containing an osu!standard v14 difficulty.")
        except Exception as exc:
            messagebox.showerror("Cannot open target", describe_error(exc))
            return
        self.target.set(str(path))
        self.levels.configure(values=names)
        self.difficulty.set(selected)
        self.refresh()

    def generate(self):
        if self.busy or not self.target.get() or not self.model.get():
            return
        parent = filedialog.askdirectory(title="Choose where to save the output folder")
        if not parent:
            return
        target = Path(self.target.get())
        difficulty = self.difficulty.get()
        if difficulty == DRAFT:
            difficulty = None
        elif target.is_file() and target.suffix.lower() in (".mp3", ".ogg", ".wav"):
            target = target.parent / difficulty
        stem = target.stem + "-hitsounded"
        output = Path(parent) / stem
        i = 1
        while output.exists():
            i += 1
            output = Path(parent) / f"{stem}-{i}"
        model = self.model.get()
        self.busy = True
        self.last = None
        self.status.set("Generating…")
        self.refresh()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.append(f"Source: {target}")
        self.append(f"Difficulty: {difficulty or DRAFT}")

        def worker():
            try:
                apply(model, target, difficulty, output, lambda s: self.messages.put(("log", s)))
                self.messages.put(("done", output))
            except Exception as exc:
                self.messages.put(("error", describe_error(exc)))

        Thread(target=worker, daemon=True).start()

    def append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def poll(self):
        while not self.messages.empty():
            kind, value = self.messages.get()
            if kind == "log":
                self.append(value)
            else:
                self.busy = False
                if kind == "done":
                    self.last = value
                    self.status.set("Complete")
                else:
                    self.append(f"Error: {value}")
                    self.status.set("Failed")
                self.refresh()
        self.root.after(100, self.poll)

    def open_output(self):
        if sys.platform == "win32":
            os.startfile(self.last)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(self.last)])


if __name__ == "__main__":
    root = tk.Tk()
    Studio(root)
    root.mainloop()
