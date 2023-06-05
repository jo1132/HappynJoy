echo "KEMDy_preprocessing processing"
#python KEMDy_preprocessing.py # KEMDy19, KEMD20 데이터셋 읽고, 저장
echo "Done!"

echo "external_data1_my_preprocessing processing"
#python external_data1_my_preprocessing.py # 감정 분류를 위한 대화 음성 데이터셋 전처리
echo "Done!"

echo "external_data2_my_preprocessing processing"
python external_data2_my_preprocessing.py # 감정분류용 데이터셋 데이터셋 전처리
echo "Done!"

echo "Data_Balancing processing"
python Data_Balancing.py # 음성데이터가 존재하지 않는 데이터 정리 및 train, test 분리
echo "Done!"