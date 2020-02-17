import requests
import operator

def emotion_classify(text):

    key = "9ab5c5c0-4ec1-11ea-a33e-e7463c4a8c877ddbdd71-0f78-42ab-a0a2-af71c71198a9"
    url = "https://machinelearningforkids.co.uk/api/scratch/"+ key + "/classify"
    lines = []
    line_len = int(len(text)/2)
    lines.append(text)
    lines.append(text[:line_len])
    lines.append(text[line_len:])


    responseDataFinal = {"happy":0 , "angry":0, "sad":0}
    for line in lines :
        response = requests.get(url, params={ "data" : line })
        responseData = response.json()
        if response.ok:
            print(responseDataFinal)
            responseDataFinal = add_dict(responseDataFinal, responseData)
        else:
            response.raise_for_status()

   
    topMatch = max(responseDataFinal.items(), key=operator.itemgetter(1))[0]
    if topMatch == "happy":
        return "행복하게"
    if topMatch == "sad":
        return "슬프게"
    if topMatch == "angry":
        return "화가 나게"


def add_dict(responseDataFinal, responseDatas):
    for responseData in responseDatas:
        life = responseData["class_name"]
        confidence = responseData["confidence"]
        responseDataFinal[life] = responseDataFinal.get(life) + confidence
    return responseDataFinal
