import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`.")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`.")



# 토크나이저 및 BERT 모델 로드
model_name = 'monologg/koelectra-base-v3-discriminator'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3,output_hidden_states = True)
checkpoint_path = "weight/checkpoint.bin"

model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
model.eval() 

# CUDA 사용 가능 여부 확인 (가능하면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
"""
def process_input(user_input):
    # 토큰화 및 인코딩
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    print(inputs)
    # 모델에 입력하여 분류 결과 얻기
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
    
    # 예측된 라벨 출력
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    
    return predicted_label
"""
def process_input(user_input):
    # 토큰화 및 인코딩
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 모델에 입력하여 분류 결과 얻기
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        logits = outputs.logits  # SequenceClassifierOutput 객체에서 logits 추출
    
    # 예측된 라벨 출력
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    
    return predicted_label

"""
def get_sentence_embedding(sentence):
    # 입력 문장을 토크나이징
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    
    # BERT 모델을 사용해 임베딩 추출
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 모델 출력이 단일 텐서일 경우, 첫 번째 토큰의 임베딩을 사용
    cls_embedding = outputs[:, :]
    
    return cls_embedding
"""
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # 마지막 레이어의 hidden state 추출
    hidden_states = outputs.hidden_states  # 이제 hidden_states는 None이 아님
    last_hidden_state = hidden_states[-1]  # 마지막 레이어의 hidden state 사용

    # [CLS] 토큰의 임베딩 추출
    cls_embedding = last_hidden_state[:, 0, :]

    return cls_embedding


"""
def calculate_cosine_similarity(sentence1, sentence2):
    # 문장 임베딩 계산
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    
    # 코사인 유사도 계산
    similarity = cosine_similarity(embedding1, embedding2)
    
    return similarity[0][0]  
"""
def calculate_cosine_similarity(sentence1, sentence2):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    
    similarity = cosine_similarity(embedding1.cpu().numpy(), embedding2.cpu().numpy())
    
    return similarity[0][0]