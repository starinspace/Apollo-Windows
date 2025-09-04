# Apollo-Windows 🌌

This is installation tutorial for using [Apollo: Band-sequence Modeling for High-Quality Audio Restoration](https://github.com/JusperLee/Apollo) on Windows.

**Tutorial for installing Apollo on Windows**

> ⚠️ **Note:** The `inference2.py` file has been re-written to work on Windows systems.

## Prerequisites

Before starting, make sure you have:

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [Git](https://git-scm.com/)

---

## Installation

### 1. Clone the original Apollo repository

```cmd
git clone https://github.com/JusperLee/Apollo
cd Apollo
```

### 2. Copy Windows-specific files

```cmd
git clone https://github.com/starinspace/Apollo-Windows.git
copy Apollo-Windows\look2hear_win.yml .
copy Apollo-Windows\inference2.py .
copy Apollo-Windows\run.bat .
rmdir /s /q Apollo-Windows
```

### 3. Create a Conda environment

```cmd
conda env create -f look2hear_win.yml -n look2hear_win
```

### 4. Activate a Conda environment

```cmd
conda activate look2hear_win
```

---

## 5. Install PyTorch based on your GPU

Check your CUDA version:

```cmd
nvidia-smi | findstr "CUDA Version"
```

Then install the correct PyTorch version:

* 🟢 **CUDA 12.6**

```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

* 🟢 **CUDA 12.8**

```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

* 🟢 **CUDA 12.9**

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

---

## 6. Install additional dependencies

Install Hugging Face Hub (optional):

```cmd
pip install huggingface-hub
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

Install any missing packages from the `.yml` file:

```cmd
pip install omegaconf
conda install -c conda-forge ffmpeg
pip install "numpy<2"
pip install soundfile
pip install pydub
conda install -c conda-forge sox
```

---

## 7. Clone the model

```cmd
git clone https://huggingface.co/JusperLee/Apollo
```

---

## 8. Create input/output folders

```cmd
mkdir input
mkdir output
```

---

✅ You are now ready to run Apollo! Simply place your audio files in the `input` folder, run `run.bat`, and the processed files will appear in the `output` folder.

---

Apollo licensed under [CC-BY-SA 4.0 International](https://github.com/JusperLee/Apollo/blob/main/LICENSE)

