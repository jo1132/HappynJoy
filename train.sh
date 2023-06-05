# 멀티모달 교사 모델 훈련
python train_crossattention.py --model_name total_multimodal_teacher

# 교사모델을 활용해, 데이터셋에 증류 데이터(Softmax) 추가
#--teacher_name 옵션으로 MultiModal 교사모델의 이름_epoch수를 입력한다.
#--data_path 옵션으로 softmax 데이터를 추가할 기존 데이터셋의 경로를 입력 (기본값, "data/train_preprocessed_data.json")
python Distill_knowledge.py --teacher_name total_multimodal_teacher_epoch29 

# miniconfig.py 를 수정해서 Epoch를 포함한 하이퍼파라미터 변경
# 멀티모달 학생 모델 지식증류 훈련i
python KD_train_crossattention.py --model_name total_multimodal_student
# 문자모달 학생 모델 지식증류 훈련
python KD_train_crossattention.py --model_name total_text_student --text_only True 
# 음성모달 학생 모델 지식증류 훈련
python KD_train_crossattention.py --model_name total_audio_student --audio_only True
