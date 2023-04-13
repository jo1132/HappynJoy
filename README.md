# HappynJoy
## Environment
```
python version : python3.9
OS type : WSL
requires packages: {
      'numpy==1.22.3',
      'pandas==1.4.2',
      'torch==1.11.0+cu113',
      'torchaudio==0.11.0+cu113',
      'scikit-learn',
      'transformers==4.18.0',
      'tokenizers==0.12.1',
      'soundfile==0.10.3.post1'
}
```
## Run
### Environment Setting
```
apt-get update && apt-get install -y
apt install ffmpeg -y
pip install numpy==1.22.3 pandas==1.4.2 scikit-learn transformers==4.18.0 tokenizers==0.12.1 soundfile==0.10.3.post1
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
### Preprocessing
```bash
python prepocessing.py
```
### Train
```bash
python train_crossattention.py
```
### Test
```bash
# pt file will generate  in ckpt after train

```