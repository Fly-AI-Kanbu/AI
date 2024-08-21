from pydantic import BaseModel
from typing import List

# Pydantic 모델 정의
class Message(BaseModel):
    content: str

class CompletionResponse(BaseModel):
    response: str

class TextInput(BaseModel):
    text: str

class TextInputs(BaseModel):
    inputs: List[TextInput]