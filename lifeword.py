import requests
import operator

def life_classify(text):

    key = "3a1d2700-4ece-11ea-a33e-e7463c4a8c87d79d1b60-01b0-4d7d-8184-8c896c2391bd"
    url = "https://machinelearningforkids.co.uk/api/scratch/"+ key + "/classify"

    lines = []
    line_len = int(len(text)/2)
    lines.append(text)
    lines.append(text[:line_len])
    lines.append(text[line_len:])

    responseDataFinal = {"character":0 , "love":0, "job":0, "friend":0, "money":0, "major":0}
    for line in lines :
        response = requests.get(url, params={ "data" : line })
        responseData = response.json()       
        if response.ok:
            responseDataFinal = add_dict(responseDataFinal, responseData)
        else:
            response.raise_for_status()

    topMatch = max(responseDataFinal.items(), key=operator.itemgetter(1))[0]

    if topMatch == "character":
        return "성격"
    elif topMatch == "love":
        return "이성관계"
    elif topMatch == "job":
        return "취업"
    elif topMatch == "friend":
        return "친구관계"
    elif topMatch == "money":
        return "돈"
    elif topMatch == "major":
        return"전공"


def add_dict(responseDataFinal, responseDatas):
    for responseData in responseDatas:
        life = responseData["class_name"]
        confidence = responseData["confidence"]
        responseDataFinal[life] = responseDataFinal.get(life) + confidence
    return responseDataFinal
