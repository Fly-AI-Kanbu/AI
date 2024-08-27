from fastapi import FastAPI
from pydantic import BaseModel
import openai

app = FastAPI()

# OpenAI API 키 설정

# 사용자 대화 상태를 저장할 딕셔너리
session_storage = {}

# Fine-tuning된 모델 이름
fine_tuned_model_name = "ft:gpt-4o-mini-2024-07-18:personal::9xpGLOhP"  # OpenAI에서 제공한 Fine-tuned 모델 ID

class ChatRequest(BaseModel):
    session_id: str
    message: str
    
@app.post("/chat")
async def chat(chat_request: ChatRequest):
    session_id = chat_request.session_id
    user_input = chat_request.message  # 이 경우 주제입니다.

    # 세션이 없을 경우 초기화
    if session_id not in session_storage:
        session_storage[session_id] = []

        # 사용자 주제를 첫 대화로 설정
        initial_prompt = {
            "role": "system",
            "content": f"당신은 친구같은 한국어 선생님입니다. 주제는 '{user_input}'입니다. 제가 어색하게 말한다면 고쳐주고, 자연스럽다면 넘어가는 형식으로, 고쳐준 부분이 있다면 그 부분에 대한 한국어 발음과 영어 번역본을 알려주고, 고쳐줄 부분이 없다면 그 답변을 참고해 다음 대화 주제를 보내줘."
        }
        session_storage[session_id].append(initial_prompt)

    # Fine-tuning된 모델로 API 호출
    response = await openai.ChatCompletion.acreate(
        model=fine_tuned_model_name,
        messages=session_storage[session_id]
    )

    # 모델의 응답을 대화 문맥에 추가
    bot_reply = response['choices'][0]['message']['content']
    session_storage[session_id].append({"role": "assistant", "content": bot_reply})

    # 응답 반환
    return {"reply": bot_reply}

@app.post("/chat2")
async def chat2(chat_request: ChatRequest):
    session_id = chat_request.session_id
    user_input = chat_request.message

    # 세션이 없을 경우 초기화
    if session_id not in session_storage:
        session_storage[session_id] = []

        # 초기 프롬프트 설정
        initial_prompt = {
            "role": "system",
            "content": "당신은 친구같은 한국어 선생님입니다. 제가 어색하게 말한다면 고쳐주고, 자연스럽다면 넘어가는 형식으로, 고쳐준 부분이 있다면 그 부분에 대한 한국어 발음과 영어 번역본을 알려주고, 고쳐줄 부분이 없다면 그 답변을 참고해 다음 대화 주제를 보내줘."
        }
        session_storage[session_id].append(initial_prompt)
    
    # 대화 문맥에 사용자 입력 추가
    session_storage[session_id].append({"role": "user", "content": user_input})

    # Fine-tuning된 모델로 API 호출
    response = await openai.ChatCompletion.acreate(
        model=fine_tuned_model_name,
        messages=session_storage[session_id]
    )

    # 모델의 응답을 대화 문맥에 추가
    bot_reply = response['choices'][0]['message']['content']
    session_storage[session_id].append({"role": "assistant", "content": bot_reply})

    # 응답 반환
    return {"reply": bot_reply}

@app.post("/chat3")
async def chat3(chat_request: ChatRequest):
    print('일루오긴함')
    session_id = chat_request.session_id
    user_input = chat_request.message

    # 세션이 없을 경우 초기화
    if session_id not in session_storage:
        session_storage[session_id] = []

        # 초기 프롬프트 설정
        initial_prompt = {
            "role": "system",
            "content": "당신은 친구같은 한국어 선생님입니다. 제가 어색하게 말한다면 고쳐주고, 자연스럽다면 넘어가는 형식으로, 고쳐준 부분이 있다면 그 부분에 대한 한국어 발음과 영어 번역본을 알려주고, 고쳐줄 부분이 없다면 그 답변을 참고해 다음 대화 주제를 보내줘."
        }
        session_storage[session_id].append(initial_prompt)
    print(f'user message : {user_input}')
    # 대화 문맥에 사용자 입력 추가
    session_storage[session_id].append({"role": "user", "content": user_input})

    # Fine-tuning된 모델로 API 호출
    response = await openai.ChatCompletion.acreate(
        model=fine_tuned_model_name,
        messages=session_storage[session_id]
    )
    print(f'gpt message : {response['choices'][0]['message']['content']}')
    # 모델의 응답을 대화 문맥에 추가
    bot_reply = response['choices'][0]['message']['content']
    session_storage[session_id].append({"role": "assistant", "content": bot_reply})

    # 응답 반환
    return {"reply": bot_reply}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2937)
