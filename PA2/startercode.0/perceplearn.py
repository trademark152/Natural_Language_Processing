# use this file to learn perceptron classifier 
# Expected: generate vanillamodel.txt and averagemodel.txt

""" Import necessary library"""
from __future__ import division
import re  # regular expression operations
import os
import fnmatch
import sys


"""Specify classes in the corpus as global variables"""
POSITIVE = 'positive'
NEGATIVE = 'negative'
DECEPTIVE = 'deceptive'
TRUTHFUL = 'truthful'

POS_DEC = 'positive_deceptive'
NEG_DEC = 'negative_deceptive'
POS_TRU = 'positive_truthful'
NEG_TRU = 'negative_truthful'

classes = [POS_DEC, POS_TRU, NEG_DEC, NEG_TRU]

stopWords = ["each","has", "had", "having", "do", "does", "did", "doing", "few", "more", "most", "other", "some", "such", "no","about", "against", "between", "into", "through", "during", "before","i", "me", "my", "myself", "we", "our", "ours","and", "but", "if",  "so", "than", "too", "very", "s", "t", "can", "will", "just", "or", "because", "as", "until", "while", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",  "of", "at", "by", "for", "with",  "after", "above","her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "have",  "a", "an", "the",  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",  "nor",  "am", "is", "are", "was", "were", "be", "been", "being","not", "only", "own", "same","don", "should", "now"]

""" Utility function"""
# create a document class:
class Document:
    # initialize a document by text and its true label
    def __init__(self, txt, trueLabel, documentName):
        self.txt = txt
        self.documentName = documentName
        self.trueLabel = trueLabel
        # create a dictionary to store word Frequency
        self.wordFreq = self.wordCount()

    # funtion to obtain the text content of the document
    def getTxt(self):
        return self.txt
    
    # function to get word frequency of a document
    def getWordFreq(self):
        return self.wordFreq

    # function to get the true label of a document
    def getTrueLabel(self):
        return self.trueLabel

    # function to get the true label of a document
    def getName(self):
        return self.documentName

    # function to count word of that document
    def wordCount(self):
        # initialize a dict: {word:wordFreq}
        wordFrequency = {}
        for word in self.txt:
            # exclude word in stop words
            if word not in stopWords:
                # "get" allows extract value of the label "word", if word is not in the dict then default it to 0
                wordFrequency[word] = wordFrequency.get(word, 0) + 1
        return (wordFrequency)

# Function to convert 2 class from text to -1 or 1:
# input: class 1 and class 2 in text format
# output: return 
# Truthful and positive are assigned 1
# Negative and deceptive are assigned -1
def trueLabel(label1, label2):
    trueLabel = []
    if label1 == TRUTHFUL:
        trueLabel.append(1)
    else:
        trueLabel.append(-1)

    if label2 == POSITIVE:
        trueLabel.append(1)
    else:
        trueLabel.append(-1)

    return trueLabel

# total size of the vocab: input is a dict
def getDictSize(vocab):
    output = 0
    for word in vocab:
        output += vocab[word]
    return output

# add a token list to an existing dictionary
def addTokensToDict(dict, tokenList):
    # initial output is the existing dict
    newDict = dict

    # loop through token
    for token in tokenList:
        # check if token is not blank
        if token != '':
            # if token is already in dict
            if token in newDict:
                newDict[token] += 1
            # if not, add to dict
            else:
                newDict[token] = 1
    return newDict

# remove all punctuations from a text
def removePuncs(text):
    # to do a regex substitution: remove all except letter, number and inherent white spaces
    return re.sub(r'[^a-zA-Z0-9 ]', r'', text)

# remove words from stopWords
def removeStops(tokenList):
    output = []
    # loop through each token to see if it is in the drop list
    for token in tokenList:
        if token in stopWords:
            # do nothing
            continue
        else:
            # return token not in stopWords
            output.append(token)
    return output

# Function to remove punctuation from a token list []
def removePunctuations(tokenList):
    output = []
    # loop through each token
    for token in tokenList:
        # to do a regex substitution: remove all except letter, number and inherent white spaces
        token = re.sub('[^a-zA-Z0-9\n\-]', '', token)
        output.append(token)
    return output

# to perform tokenization and processing of a file 
# input: fileName
# output: clean list of tokens in that file
def processFile(fileName):
    # open file
    fileObj = open(fileName, "r")

    # read file
    fileContent = fileObj.read()

    # lower case all content
    fileContent = fileContent.lower()

    # split to tokens
    tokens = fileContent.split()

    # remove punctuations
    tokensNoPunc = removePunctuations(tokens)

    # remove stop word
    tokensFinal = removeStops(tokensNoPunc)

    # close the file
    fileObj.close()
    return tokensFinal


"""define vanilla perceptron model"""
# Input:
# set of documents docs
# number of iteration numIter
# weight of binary classifier 1: (T or D) {[word1:word1Weight]...
# weight of binary classifier 2: (P or N)
# bias for both classifer: [bias[0] for T/D, bias[1] for P/N]
def vanillaPerceptron(docs, numIter, weightsTDVan, weightsPNVan, biasVan):
    # loop through iterations
    for i in range(numIter):
        # loop through doc in docs:
        for doc in docs:
            # get word frequency of each doc
            wordCountDoc = doc.getWordFreq()
            # print(wordCountDoc)

            # get true label of that doc [T or D, P or N]
            labelDoc = doc.getTrueLabel()
            # print(labelDoc)

            # initialize weight sum
            weightSumTDVan = 0
            weightSumPNVan = 0

            # loop through word in word counter
            for word in wordCountDoc.keys():
                # exclude word in stop word list
                if word in stopWords:
                    continue

                # initialize if word has not shown up in weight dict's keys
                if word not in weightsTDVan:
                    weightsTDVan[word] = 0
                if word not in weightsPNVan:
                    weightsPNVan[word] = 0

                # update weight sum. dict.get helps you return 0 if word is not in the dict
                # weightSum = weight * wordFrequency
                weightSumTDVan += weightsTDVan.get(word, 0) * wordCountDoc[word]
                weightSumPNVan += weightsPNVan.get(word, 0) * wordCountDoc[word]

            # if misclassified, update weight and bias if yTrue*yPred<=0
            # For Truthful or Deceptive classifier
            if labelDoc[0] * (weightSumTDVan + biasVan[0]) <= 0:
                # loop through words of that document to update weight and bias
                for word in wordCountDoc.keys():
                    # only use words outside stop word list
                    if word not in stopWords:
                        # update weight +=
                        weightsTDVan[word] = weightsTDVan.get(word, 0) + (labelDoc[0] * wordCountDoc[word])

                # update bias outside the loop for words
                biasVan[0] = biasVan[0] + labelDoc[0]

            # Similar for Positive or negative classifier
            if labelDoc[1] * (weightSumPNVan + biasVan[1]) <= 0:
                for word in wordCountDoc.keys():
                    if word not in stopWords:
                        weightsPNVan[word] = weightsPNVan.get(word, 0) + (labelDoc[1] * wordCountDoc[word])
                biasVan[1] = biasVan[1] + labelDoc[1]

    # print("Final weight for binary classifier Truthful/Deceptive: ", weightsTDVan)
    # print("Final weight for binary classifier Positive/Negative: ", weightsPNVan)
    # print("Biases for both binary classifiers: ", biasVan)

"""define average perceptron model"""
# Input:
# set of documents docs
# number of iteration numIter
# weight of binary classifier 1: (T or D) {[word1:word1Weight]...
# weight of binary classifier 2: (P or N)
# bias for both classifer: [bias[0] for T/D, bias[1] for P/N]
# SEE CHAPTER 4 DAUME BOOK algorithm 7
def averagePerceptron(docs, numIter, weightsTDAvg, weightsPNAvg, biasAvg):
    # Initialize example counter to 1
    cTD = 1
    cPN = 1
    
    # Initialize cached bias
    betaTD = 0
    betaPN = 0
    
    # Initialize cached weights as dictionary
    uTD = {}
    uPN = {}
    
    # loop through iterations similar to vanilla
    for i in range(numIter):
        for doc in docs:
            wordCountDoc = doc.getWordFreq()
            labelDoc = doc.getTrueLabel()
            weightSumTDAvg = 0
            weightSumPNAvg = 0

            for word in wordCountDoc.keys():
                if word in stopWords:
                    continue

                if word not in weightsTDAvg:
                    weightsTDAvg[word] = 0

                if word not in weightsPNAvg:
                    weightsPNAvg[word] = 0

                weightSumTDAvg += weightsTDAvg.get(word, 0) * wordCountDoc[word]
                weightSumPNAvg += weightsPNAvg.get(word, 0) * wordCountDoc[word]

            # if misclassify in T/D
            if labelDoc[0] * (weightSumTDAvg + biasAvg[0]) <= 0:
                for word in wordCountDoc.keys():
                    if word not in stopWords:
                        # update weight w <- w+y*x
                        weightsTDAvg[word] = weightsTDAvg.get(word, 0) + (labelDoc[0] * wordCountDoc[word])

                        # update cached weights u <- u+y*c*x
                        uTD[word] = uTD.get(word, 0) + float(labelDoc[0] * cTD * wordCountDoc[word])

                # update cached biases (beta <- beta + y*c) and bias
                biasAvg[0] = biasAvg[0] + labelDoc[0]
                betaTD = betaTD + float(labelDoc[0] * cTD)

            # increment counter regardless of update to keep moving average consistent
            cTD = cTD + 1

            # if misclassify in P/N
            if labelDoc[1] * (weightSumPNAvg + biasAvg[1]) <= 0:
                for word in wordCountDoc.keys():
                    if word not in stopWords:
                        weightsPNAvg[word] = weightsPNAvg.get(word, 0) + (labelDoc[1] * wordCountDoc[word])
                        uPN[word] = uPN.get(word, 0) + float(labelDoc[1] * cPN * wordCountDoc[word])
                biasAvg[1] = biasAvg[1] + labelDoc[1]
                betaPN = betaPN + float(labelDoc[1] * cPN)
            cPN = cPN + 1

    # for each binary classifier:
    # return averaged weights wAvg = w-1/c*u; 
    for word in weightsTDAvg.keys():
        weightsTDAvg[word] = weightsTDAvg.get(word, 0) - (float(1 / cTD) * uTD.get(word, 0))
    # return averaged bias:  biasAvg = bias - 1/c*beta
    biasAvg[0] = biasAvg[0] - float(1 / cTD) * betaTD

    for word in weightsPNAvg.keys():
        weightsPNAvg[word] = weightsPNAvg.get(word, 0) - (float(1 / cPN) * uPN.get(word, 0))
    biasAvg[1] = biasAvg[1] - float(1 / cPN) * betaPN

"""MAIN"""
# input_path is the path to the training files
def main(input_path):

    # define the base path
    # print("Root directory of the training data: ", input_path)

    #paths ot the training data
    # positive deceptive
    pos_dec = input_path+"/positive_polarity/deceptive_from_MTurk/"
    # print(pos_dec)

    # positive truthful
    pos_tru = input_path+"/positive_polarity/truthful_from_TripAdvisor/"

    # negative deceptive
    neg_dec = input_path +"/negative_polarity/deceptive_from_MTurk/"

    # negative truthful
    neg_tru = input_path+"/negative_polarity/truthful_from_Web/"

    # specify path directory for each class:
    path_directory = { POS_DEC: [pos_dec],
                       NEG_DEC: [neg_dec],
                       NEG_TRU: [neg_tru],
                       POS_TRU: [pos_tru]}
    # print("Path Directory of each class: ", path_directory)

    """ PERCEPTRON MODEL"""
    # initiate Vanilla perceptron model's parameters
    weightsTDVan = {}
    weightsPNVan = {}
    biasVan = [0, 0]

    """ Average MODEL"""
    # initiate Vanilla perceptron model's parameters
    weightsTDAvg = {}
    weightsPNAvg = {}
    biasAvg = [0, 0]

    # Number of iterations
    iterations = 20

    # list of all docs
    alldoc = []

    # Loop through each class and corresponding directory
    for label, directory in path_directory.items():

        # loop through each directory path belong to each class
        for path in directory:
            # print("Building vocab for ", label,"in path ", path)

            # loop through each file in the path
            for root, directoryName, fileNames in os.walk(path):

                # filter out file that ends with .txt (file name pattern matching)
                for fileName in fnmatch.filter(fileNames, '*.txt'):
                    # reconstruct file name
                    fileName = os.path.join(root, fileName)
                    # print("currently reading: ", fileName)

                    # process the file to get the list of tokens
                    fileTokens = processFile(fileName)
                    # print(fileTokens)

                    # Get true label for that file
                    if(label == POS_DEC):
                        # extract class
                        labelTD = DECEPTIVE
                        labelPN = POSITIVE

                    elif(label == NEG_DEC):
                        labelTD = DECEPTIVE
                        labelPN = NEGATIVE

                    elif(label == POS_TRU):
                        labelTD = TRUTHFUL
                        labelPN = POSITIVE

                    elif(label == NEG_TRU):
                        labelTD = TRUTHFUL
                        labelPN = NEGATIVE

                    # create a document with text content and 2 true classes
                    doc = Document(fileTokens, trueLabel(labelTD, labelPN), fileName)

                    # add doc to set of documents prepared to be trained
                    alldoc.append(doc)

    # Run the model:
    vanillaPerceptron(alldoc, iterations, weightsTDVan, weightsPNVan, biasVan)
    averagePerceptron(alldoc, iterations, weightsTDAvg, weightsPNAvg, biasAvg)

    """WRITE OUTPUT"""
    # write to perceptron model file
    model_file = "vanillamodel.txt"
    avg_model_file = "averagedmodel.txt"

    # open file to write
    fileOut = open(model_file, 'w')

    # write content
    fileOut.write(str(weightsTDVan))
    fileOut.write('\n')
    fileOut.write(str(weightsPNVan))
    fileOut.write('\n')
    fileOut.write(str(biasVan))
    fileOut.write('\n')
    fileOut.write(str(stopWords))
    fileOut.write('\n')
    
    # close the file
    fileOut.close()

    # open file to write
    fileOut2 = open(avg_model_file, 'w')

    # write content
    fileOut2.write(str(weightsTDAvg))
    fileOut2.write('\n')
    fileOut2.write(str(weightsPNAvg))
    fileOut2.write('\n')
    fileOut2.write(str(biasAvg))
    fileOut2.write('\n')
    fileOut2.write(str(stopWords))
    fileOut2.write('\n')

    # close the file
    fileOut2.close()

""" RUN THE PROGRAM"""

# local folder: C:\Users\trade\Dropbox\Classes\Spring 2019\CSCI544\HW\PA\PA2\startercode.0>

# For local path
# inPath = os.path.dirname(os.path.realpath(__file__)) + "\\op_spam_training_data\\"

# for command line invoke: python ./perceplearn.py ./op_spam_training_data
inPath = str(sys.argv[-1])
# print("inPath: ", inPath)

# run the main program
main(inPath)