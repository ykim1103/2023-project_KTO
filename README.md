# 2023-project_KTO
KTO(Korea Tourism Organization), 한국관광공사 데이터랩 운영

#### modules : py에 필요한 공통 모듈폴더
   - data_loader.py : 학습 데이터 가공 파일
   - dbconn.py : DB에서 필요한 데이터 로딩 파일
   - loggers.py : 로그생성 파일
   - lstm.py : LSTM모델 파일
   - request_info.py : 학습요청상태 및 모델 사용여부 확인 파일
   - retrain_test_c.py : 클러스터별 재학습 및 테스트 파일(시군구용)
   - retrai_test.py : 재학습 및 테스트 파일(시도용)
#### sns_sgg_predict.py : SNS_시군구 예측파일(75일 예측)
#### sns_sgg_train.py : SNS_시군구 학습파일
#### sns_sgg_valid.py : SNS_시군구 검증파일(실제데이터 수급 후 예측한 결과와 비교 후 검증)
#### sns_sido_predict.py : SNS_시도 예측파일(75일 예측)
#### sns_sido_train.py : SNS_시도 학습파일
#### sns_sido_valid.py : SNS_시도 검증파일(실제데이터 수급 후 예측한 결과와 비교 후 검증)
---------------------------------------------------------------------------------------------
