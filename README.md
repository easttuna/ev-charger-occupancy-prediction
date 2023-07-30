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
    - (1) MultiSeqBase : close history & weekly history 두개의 시퀀스 + 시간관련 변수 사용
    - (2) MultiSeqNormal: (1) + 가공없는 충전소 특성 벡터
    - (3) MultiSeqUmap: (1) + UMAP 차원 축소 충전소 특성 벡터
    - (4) MultiSeqUmapEmb: (3)에서 UMAP 특성벡터로 임베딩 초기화해서 추가 학습되도록함
    - (5) MultiSeqNormalUmapemb: (4) + 가공없는 충전소 특성벡터
    - (6) MultiSeqUmapEmbGating: (4) + 충전소 특성에 따라 close history, weekly history의 가중치 부여하는 게이팅 함수 추가
    - (7) MultiSeqNormalUmapGating: (5)에 gating 개념 추가
