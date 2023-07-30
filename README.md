# ev-charger-occupancy-prediction
EV-charger occupancy prediction with DNN


## 최종 작업물
### notebooks
- GIS 통해 가공한 데이터 + 테이블 데이터 전처리 과정
- STEP
  - 01_data_selection_preprocess_multiclass.ipynb
  - 02_attribute_addition_multiclass_final.ipynb
  - 03_result_discussion.ipynb
 
### python Modules
- models.py
  - 본격적인 개발 이전, 아이디어를 간단히 구현하 프로토 타입 모델들을 시계열 예측의 단순 모델 (naive, mean)과 비교해봄
  - 최종 분석에는 사용되지 않으므로 살펴보지 않아도 무방
- basemodels_multiclass.py
  - 최종 평가에 사용한 모델 클래스를 모아놓은 모듈 (basemodels 라는 네이밍이 맞지 않지만..)
  - 모델명 용어설명
    - (1) MultiSeqBase : realtime & historic 두개의 시퀀스 + 시간관련 변수 사용
    - (2) MultiSeqNormal: (1) + 가공없는 충전소 특성 벡터
    - (3) MultiSeqUmap: (1) + UMAP 차원 축소 충전소 특성 벡터
    - (4) MultiSeqUmapEmb: (3)에서 UMAP 특성벡터로 임베딩 초기화해서 추가 학습되도록함
    - (5) MultiSeqNormalUmapemb: (4) + 가공없는 충전소 특성벡터
    - (6) MultiSeqUmapEmbGating: (4) + 충전소 특성에 따라 close history, weekly history의 가중치 부여하는 게이팅 함수 추가
    - (7) MultiSeqNormalUmapGating: (5)에 gating 개념 추가
- dataset.py
  - r: realtime(;close) sequence 변수
  - h: historyc(;weekly) sequence 변수
  - t: 시간관련 변수 (타임인덱스, 요일 등)
  - s: 충전소 특성 변수
  - y: 표적 변수
- preprocess.py: 충전기 이용이력 로그테이블을 각 충전소별x타임인덱스별 점유상태로 변환하는 함수
- utils.py
  - preprocess에서 변환한 시간인덱스별 점유상태를 모델 입력에 맞게 가공(realtime sequence, historic sequence, target sequence)
  - time_features: 시간 관련 변수 복제. 다른 충전소도 같은 시점엔 동일한 시간변수를 가지므로 np.repeat 으로 복사
  - station_features: 충전소 특성 변수 복제. 각 시점에 대한 충전소 데이터가 반복되므로, np.tile로 복사
  - linear_split: 시간축으로 데이터 분할
  - station_wise_split: 학습/평가 데이터에 충전소간 교집합이 없도록 분할
  - EvcFeatureGenerator: 위 외부함수와 내부 함수, 속성 등 종합하여 모델 입력 데이터 추출함
    - historic_seq_smoothing
      - weekly 변수 이용시, 정확히 일주일전의 동일 타임인덱스의 점유상태가 일반적인 점유상태를 표현한다고 보기 어려움
      - 우연에 의해 점유/비점유 일 수 있으므로, 1시간 범위로 뭉개주어 해당 시간대의 일반적인 점유상태를 반영하도록 함
