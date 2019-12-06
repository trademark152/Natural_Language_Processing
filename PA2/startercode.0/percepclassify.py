import os
import fnmatch
import re
import sys
import glob


"""Initialization"""
# all correct classes
POSITIVE = 'positive'
NEGATIVE = 'negative'
DECEPTIVE = 'deceptive'
TRUTHFUL = 'truthful'
POS_DEC = 'positive_deceptive'
NEG_DEC = 'negative_deceptive'
POS_TRU = 'postiive_truthful'
NEG_TRU = 'negative_truthful'



"""MAIN"""
def main(model_path, input_path):
    """ Utility function"""

    # function to count word from a text (excluding words from stopWordsList
    def wordCount(text, stopWordsList):
        wordCounter = {}
        for word in text:
            if word not in stopWordsList:
                wordCounter[word] = wordCounter.get(word, 0) + 1
        return (wordCounter)

    # remove words from stopWordsList
    def removeStops(tokenList):
        output = []
        # loop through each token to see if it is in the drop list
        for token in tokenList:
            if token in stopWordsList:
                # do nothing
                continue
            else:
                # return token not in stopWordsList
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

    # to perform tokenization and processing of a file and return list of tokens
    def processFile(fileName):
        # open file
        fileObj = open(fileName, "r")

        # read file
        fileContent = fileObj.read()

        # lower case all content
        fileContent = fileContent.lower()

        # split to tokens
        tokens = fileContent.split()

        # # store file id
        # fileID = tokens[0]
        #
        # # Truthful or deceitful
        # td = tokens[1]
        #
        # # positive or negative
        # pn = tokens[2]

        # exclude the first three tokens (data)
        newTokens = tokens[3:]

        # remove punctuations
        tokensNoPunc = removePunctuations(newTokens)

        # remove stop word
        tokensFinal = removeStops(tokensNoPunc)

        # close the file
        fileObj.close()
        return tokensFinal

    # total size of the vocab: input is a dict
    def getDictSize(vocab):
        output = 0
        for word in vocab:
            output += vocab[word]
        return output

    """define vanilla perceptron model"""

    # Input:
    # label
    # line
    # weight of binary classifier 1: (T or D) {[word1:word1Weight]...
    # weight of binary classifier 2: (P or N)
    # bias for both classifer: [bias[0] for T/D, bias[1] for P/N]
    # stopWordsList
    # Output:
    # label
    def predict(labels, text, weightTD, weightPN, bias, stopWordsList):
        # count words of the text
        wordCounter = wordCount(text, stopWordsList)

        # initialize weight sum
        weightSum = [0, 0]

        # loop through words
        for word in wordCounter.keys():
            # exclude word in stop word list
            if word in stopWordsList:
                continue  # means moving to the next word in for loop without doing anything

            # initialize new word in the weight
            if word not in weightTD:
                weightTD[word] = 0
            if word not in weightPN:
                weightPN[word] = 0

            # update weight sum for each binary classifier
            weightSum[0] += weightTD.get(word, 0) * wordCounter[word]
            weightSum[1] += weightPN.get(word, 0) * wordCounter[word]

        # classify
        if (weightSum[0] + bias[0]) > 0:
            labels.append("truthful")
        else:
            labels.append("deceptive")

        if (weightSum[1] + bias[1]) > 0:
            labels.append("positive")
        else:
            labels.append("negative")

        return labels

    """ open model file to extract parameters """
    # print("Root directory of the model: ", model_path)
    model = open(model_path, 'r')
    params = model.readlines()
    model.close()

    # import data
    weightTD = eval(params[0])
    weightPN = eval(params[1])
    bias = eval(params[2])
    stopWordsList = eval(params[3])


    """ paths ot the testing data """
    # base path for testing data
    # print("Root directory of the testing data: ", input_path)

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

    """ Create output file"""
    # match output file name with the given format
    output_file = "percepoutput.txt"
    outputFile = open(output_file, "w")

    # initialize final labels [[label1 of doc1, label2 of doc1],[label1 of doc2, label2 of doc2]...]
    finalLabels = []

    # Loop through each testing directory: key = correct classification, value = [correctClass1, correctClass2]
    # get all the files' directories
    # Loop through each class and corresponding directory
    # for label, directory in path_directory.items():
    #
    #     # loop through each directory path belong to each class
    #     for path in directory:
    #         # print("Building vocab for ", label,"in path ", path)
    #
    #         # loop through each file in the path
    #         for root, directoryName, fileNames in os.walk(path):
    #
    #             # filter out file that ends with .txt (file name pattern matching)
    #             for fileName in fnmatch.filter(fileNames, '*.txt'):

    filePaths = glob.glob(os.path.join(sys.argv[-1], '*/*/*/*.txt'))
    # print("Files: ", filePaths)
    for fileName in filePaths:
        # initialize label of this file
        tempLabels = []

        # reconstruct file name
        # fileName = os.path.join(root, fileName)
        # print("currently reading: ", fileName)

        # get the tokens from file after processing
        fileTokens = processFile(fileName)

        # get the labels
        predict(tempLabels, fileTokens, weightTD, weightPN, bias, stopWordsList)

        # reconstruct output as given format
        # output = tempLabels[0] + " " + tempLabels[1] + " " + fileName + " " + label  # to know the true label
        output = tempLabels[0] + " " + tempLabels[1] + " " + fileName
        # print(output)

        # append to final results
        finalLabels.append(output)

    # write to output file
    for item in finalLabels:
        outputFile.write(item)
        outputFile.write('\n')
    # close output file
    outputFile.close()

"""RUN CLASSIFICATION"""
# local folder: C:\Users\trade\Dropbox\Classes\Spring 2019\CSCI544\HW\PA\PA2\startercode.0>

# For local path
# inPath = os.path.dirname(os.path.realpath(__file__)) + "\\op_spam_training_data\\"
# modelPath1 = os.path.dirname(os.path.realpath(__file__)) + "\\vanillamodel.txt"
# modelPath2 = os.path.dirname(os.path.realpath(__file__)) + "\\averagedmodel.txt"

# to invoke in command line: python ./percepclassify.py ./vanillamodel.txt ./op_spam_testing_data
modelPath = str(sys.argv[1])
inPath = str(sys.argv[2])
# print("modelPath: ", modelPath)
# print("inPath: ", inPath)

# RUN
main(modelPath, inPath)
# main(modelPath1, inPath)
# main(modelPath2, inPath)







