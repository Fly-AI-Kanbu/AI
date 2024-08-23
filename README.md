# AI

### Environment Setting
(1) Create your virtual environment

(2) Install requirements
```
pip install -r requirements.txt
```

(3) Download the bareun.ai from the homepage. (https://bareun.ai/download)

(4) Install kiwi 

``` 
pip install --upgrade pip 
pip install kiwipiepy
```

(5) Move your checkpoint file to "AI/weight" (we use koelectra-base-v3-discriminator)

(6) Run main.py

### Evaluation Metrics


(1) Delivery_score : 전달력 분류기를 활용 (0,1,2)

(2) Toxicity_score : 독성도 (Perspective API)

(3) cos_sim_question  : 질-답간 유사도

(4) cos_sim_answer : 답-교정문 간 유사도

(5) mlum_score : 문장 내 형태소 개수 / 문장 개수

