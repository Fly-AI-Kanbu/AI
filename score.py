
from googleapiclient import discovery
from kiwipiepy import Kiwi
from bareunpy import Tagger
import os

# Perspective API Key
client_toxic = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=os.environ['API_KEY_FOR_PERSPECTIVE'],
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

# 바른 형태소 분석기
tagger = Tagger(os.environ['API_KEY_FOR_BAREUN'], 'localhost')


# 문장 분리기
kiwi = Kiwi()


def analyze_toxicity(text: str):
    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {'TOXICITY': {}}
    }
    
    try:
        response = client_toxic.comments().analyze(body=analyze_request).execute()
        score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        return score
    except Exception as e:
        print(f"Error during toxicity analysis: {str(e)}")
        return None



 
def morph(sentence):
    sentence_num = sentence_seperate(sentence)
    morph_num = len(tagger.morphs(sentence))
    result = sentence_num // morph_num

    return result

def sentence_seperate(sentence):
    return len(kiwi.split_into_sents(sentence))
