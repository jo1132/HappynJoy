apt-get update && apt-get install -y
apt install ffmpeg -y
pip install numpy==1.22.3 pandas==1.4.2 scikit-learn transformers==4.18.0 tokenizers==0.12.1 soundfile==0.10.3.post1 moviepy
pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#cd HappynJoy
python KEMDy_my_preprocessing.py
python external_data1_my_preprocessing.py
python external_data2_my_preprocessing.py
python data_balancing.py
#python Distill_knowledge.py --model_name teacher_epoch30/test_epoch29