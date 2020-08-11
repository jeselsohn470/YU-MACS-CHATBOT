# YU-MACS-CHATBOT

A chatbot implementation with PyTorch and NLTK libraries and a Feed Forward Neural net with 2 hidden layers. 

The training data is in the intents.json file. In this file, there are different intents. For each intent, there is a tag (a class label) and there are different patterns for the tag and then a response for the tag. When a question comes in, the bot tries to classify the question into one of the tags and then returns the response for that tag. If the bot cannot classify the question into one of the tags, then it returns, "I'm sorry, I do not know the answer to that question." 

With the training data, we train the deep learning model using a concept called, "Bag of Words". In order to do this, we collect all the words from all the different patterns, and put them into a String array (called "all_words"). Then, for each pattern we create an array, called bag_of_words, with the same size as the all_words array and if the word from the pattern is included in the all_words array, we put a 1 in the associated position in the newly created bag_of_words array. Otherwise, we put a 0. In order to get from the question to the bag of words, we have to apply two NLP technique. The first one is called tokenization, which means splitting the String into meaningful units (e.g. words, characters, numbers, punctuation). The next concept we apply is stemming, which generates the root form of the words. In other words, stemming chops off the end of words. 

Example of our NLP process:

1. Tokenization:
"What financial aid packages are available?" => tokenize => ["What", "financial", "aid", "packages", "are", "available", "?"]

2. Make Everything Lower Case and Stemming:
["What", "financial", "aid", "packages", "are", "available", "?"] => lower and stem => ["what", "financ", "aid", "package", "are", "availab", "?"]

3. Exclude Punctuation:
["what", "financ", "aid", "package", "are", "availab", "?"] => ["what", "financ", "aid", "package", "are", "availab"]

4. Bag of Words:
["what", "financ", "aid", "package", "are", "availab"] => [1,0,0,1,1,0,1,0,0,1,0,1]


Explanation of the Files:

In nltk_utils.py, we use all of the NLTK utilities mentioned above (tokenization, stemming and bag of words) to get from the question the user entered to the bag of words array.

In train.py, we load the json intents file and create the training data by collecting all of the words and putting them in the all_words list. We also create a list that holds all the tags from the json file. Finally, we create a list called xy which holds both the patterns and the tags from the json file. 
After this, we create the chat data set and train the data by passing it through the neural network. Once the training is completed, the deep learning model is created and the model and data are saved in data.pth.

model.py is the code which creates the model for the chatbot. 

In chat.py, the chat function is implemented. If the model can classify the question into one of the tags with a probability of at least 80 percent, then the bot returns the response from that tag. Otherwise, the bot will return, "I'm sorry, I do not know the answer to that question." 




 
