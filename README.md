# Apollo-Windows
Tutorial for installing Apollo on Windows
Install Miniconda and Git

```cmd
git clone https://github.com/JusperLee/Apollo
cd Apollo
```
copy files from this respority into the Apollo folder
```cmd
git clone https://github.com/starinspace/Apollo-Windows.git
copy Apollo-Windows\look2hear_win.yml .
copy Apollo-Windows\inference2.py .
copy Apollo-Windows\run.bat .
rmdir /s /q Apollo-Windows
```

Create enviroment for Conda
```cmd
conda env create -f look2hear_win.yml -n look2hear_win
conda activate look2hear_win
```

Find what version of GPU you have
```cmd
nvidia-smi | findstr "CUDA Version"
```

🟢 For Cuda 12.6
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
🟢 For Cuda 12.8
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
🟢 For Cuda 12.9
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```
Install huggingface-hub (might not be needed)
```cmd
pip install huggingface-hub
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```
Install things that might not worked from yml-file
```cmd
pip install omegaconf
conda install -c conda-forge ffmpeg
pip install "numpy<2"
pip install soundfile
pip install pydub
conda install -c conda-forge sox
```
Clone the model
```cmd
git clone https://huggingface.co/JusperLee/Apollo
```
create folder for input and output
```cmd
mkdir input
mkdir output
```

