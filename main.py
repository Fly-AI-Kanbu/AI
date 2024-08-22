from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from openai import OpenAI
from dotenv import load_dotenv
from googleapiclient import discovery
from typing import List
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os, io, json
import schemas, aimodels, score

# .env 파일에서 환경 변수 로드
load_dotenv()

app = FastAPI()

client_openai = OpenAI(
    api_key = os.environ['OPENAI_API_KEY']
)

####
#####openai를 활용한 api
@app.post("/chat", response_model=schemas.CompletionResponse)
async def chat_completion(message: schemas.Message):
    try:
        # GPT-4 모델에 요청 보내기
        completion = client_openai.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::9xpGLOhP", 
            messages=[
                {"role": "system", "content": "당신은 친구같은 한국어 선생님입니다. 제가 어색하게 말한다면 고쳐주고, 자연스럽다면 넘어가는 형식으로, 고쳐준 부분이 있다면 그 부분에 대한 한국어 발음과 영어 번역본을 알려주고, 고쳐줄 부분이 없다면 그 답변을 참고해 다음 대화 주제를 보내줘."},
                {"role": "user", "content": message.content}
            ]
        )

        # 응답 반환
        return schemas.CompletionResponse(response=completion.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/stt/")
async def stt(file: UploadFile = File(...)):
    try:
        # 파일 확장자 확인
        filename = file.filename
        if not filename.lower().endswith(('.flac', '.m4a', '.mp3', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg', '.wav', '.webm')):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # 파일을 바이트로 읽고 BytesIO로 래핑
        audio_bytes = await file.read()
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename  # 파일 이름 설정

        # OpenAI API에 파일 전달
        transcript = client_openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file  # io.BytesIO 객체를 파일처럼 전달
        )

        # 텍스트 결과 반환
        return JSONResponse(content={"transcript": transcript.text})
    
    except HTTPException as e:
        raise e  # 발생한 HTTPException 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/")
async def synthesize_speech(input_text: str = Form(...), speed: float = Form(1), voice: str = Form("nova")):
    try:
        # Call the OpenAI TTS model
        response = client_openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=input_text,
            speed=speed,
            response_format='mp3'
        )

        # Use BytesIO to handle the file in memory
        audio_stream = io.BytesIO(response.content)
        audio_stream.seek(0)  # Ensure the stream is at the beginning

        # Return the mp3 file as a response
        return StreamingResponse(audio_stream, media_type='audio/mpeg', headers={
            "Content-Disposition": "attachment; filename=output.mp3"
        })

    except Exception as e:
        return {"error": str(e)}
    
####분류기####
@app.post("/bert", response_model=List)
def predict(input: schemas.TextInputs):
    try:
        results = []
        
        # 여러 텍스트 항목을 반복 처리
        for item in input.inputs:
            user_input = item.text
            # BERT 모델을 사용하여 입력 처리 (여기서는 예시로 대체)
            label = aimodels.process_input(user_input)
            results.append(label)
        
        # 결과 반환
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict")
async def predict(input: schemas.modelInputs):

    # 파일의 내용을 읽어 들입니다.
    dialogue_length = len(input.inputs)

    Delivery_score = 0
    Toxicity_score = 0
    cos_sim_question = 0
    cos_sim_answer = 0
    mlum_score= 0

    for line in input.inputs:
        # 각 줄을 JSON으로 파싱 -> {1, 2, 3}, {3, 4, 5}
        data = line
        
        turn1 = data.model1
        turn2 = data.user
        turn3 = data.model2
        
        #입력된 대화 전처리
        Delivery_score += aimodels.process_input(turn2)
        Toxicity_score += score.analyze_toxicity(turn2)
        cos_sim_question += aimodels.calculate_cosine_similarity(turn1, turn2)
        cos_sim_answer += aimodels.calculate_cosine_similarity(turn2, turn3)
        mlum_score += score.morph(turn2)
  
    result = {
        "Delivery_score" : Delivery_score / dialogue_length,
        "Toxicity_score" : Toxicity_score / dialogue_length,
        "cos_sim_question" : cos_sim_question / dialogue_length,
        "cos_sim_answer" : cos_sim_answer / dialogue_length,
        "mlum_score" : mlum_score / dialogue_length,
        }
    
    return result
