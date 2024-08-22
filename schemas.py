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


class modelInputBase(BaseModel):
    model1: str
    user: str
    model2: str

class modelInputsBase(BaseModel):
    inputs: List[modelInputBase]

class modelInput(modelInputBase):
    class Config:
        from_attributes = True

class modelInputs(modelInputsBase):
    class Config:
        from_attributes = True
