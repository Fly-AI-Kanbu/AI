import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`.")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`.")

# BERT 모델 정의 (분류 레이어 추가)
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 임베딩
        logits = self.classifier(cls_output)
        return logits

# 토크나이저 및 BERT 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 분류 레이어를 가진 새로운 모델 생성
model = BertClassifier(bert_model, num_labels=2)  # 예제에서는 2개의 클래스를 분류한다고 가정

# CUDA 사용 가능 여부 확인 (가능하면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def process_input(user_input):
    # 토큰화 및 인코딩
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 모델에 입력하여 분류 결과 얻기
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
    
    # 예측된 라벨 출력
    predicted_label = torch.argmax(logits, dim=1).cpu().item()
    
    return predicted_label
