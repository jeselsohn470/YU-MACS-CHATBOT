import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]

hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# bot_name = "MACS"
# print("Hi There!")
#
# # while True:
# #     sentence = input("You: ").lower()
# #     # sentence = "do you use credit cards?"
# #
# #     sentence = tokenize(sentence)
# #     X = bag_of_words(sentence, all_words)
# #     X = X.reshape(1, X.shape[0])
# #     X = torch.from_numpy(X).to(device)
# #
# #     output = model(X)
# #     _, predicted = torch.max(output, dim=1)
# #
# #     tag = tags[predicted.item()]
# #
# #     probs = torch.softmax(output, dim=1)
# #     prob = probs[0][predicted.item()]
# #     if prob.item() > 0.75:
# #         if tag == "goodbye":
# #             print(bot_name + ": I hope you found the answers you were looking for. Goodbye now")
# #             break
# #         for intent in intents['intents']:
# #             if tag == intent["tag"]:
# #                 if tag == "greeting":
# #                     print(bot_name + ": " + random.choice(intent['responses']))
# #                     break
# #                 print(bot_name + ": " + random.choice(intent['responses']) + "\n\nDo you have any other questions?")
# #
# #     else:
# #         officer = input(bot_name + ": I do not understand... Would you like to speak to an Admissions Officer? \nYou: ").lower()
# #
# #         sentence = tokenize(officer)
# #         X = bag_of_words(sentence, all_words)
# #         X = X.reshape(1, X.shape[0])
# #         X = torch.from_numpy(X).to(device)
# #
# #         output = model(X)
# #         _, predicted = torch.max(output, dim=1)
# #
# #         tag = tags[predicted.item()]
# #
# #         probs = torch.softmax(output, dim=1)
# #         prob = probs[0][predicted.item()]
# #
# #         if prob.item() > 0.75 and tag == "yes":
# #             print("Ok. I will connect you with an Admissions Officer shortly")
# #             break
# #         else:
# #             print(bot_name + ": Do you have any other questions then? If so, what are they?")


def predict(question):
    sentence = question.lower()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not know the answer to that question. Would you like to speak to an Admissions Officer directly?"


## for testing purposes
print(predict(input("You: ")))