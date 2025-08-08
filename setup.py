import os
from cx_Freeze import setup, Executable

build_options = {
    "packages": [
        "librosa",
        "soundfile",
        "spleeter",
        "numpy",
        "tensorflow",
        "sklearn",
    ],
    "excludes": ["tkinter"],
    "include_files": [
        (r"C:\Windows\System32\vcomp140.dll", "vcomp140.dll"),
    ],
}

setup(
    name="Voice splitter",
    options={"build_exe": build_options},
    executables=[Executable("Voice splitter script.py", base=None)],
)