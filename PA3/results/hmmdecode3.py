""" Approach"""
"""
* Need to find P(tag1,tag2,...|word1,word2,...) ~ P(word1,word2,...|tag1,tag2,...)*P(tag1,tag2...)
* Use training corpus to find emission probabilities P(word|tag) and transitional probabilities P(tag1,tag2...)
* Corpus = several lines; Line = several word|tag; Word|Tag = Word and Tag
"""

""" Hidden markov model (see problem 3 hw2) from a tagged corpus"""
"""
  - each state is a part-of-speech tag
  - each tag can emit observations, which are words
  - transition probabilities are the conditional probabilities of tags sequence
  - emission probabilities are the conditional probabilities of words given tags. 
  - The start state is the beginning of a sentence, which is not a partof-speech tag
"""

""" IMPORT LIBRARIES"""
import sys
import math

"""" DECLARE VARIABLES"""
model_file = "hmmmodel.txt"  # output file for hmm model
output_file = "hmmoutput.txt" # output file
emisProb = {}  # dict of {word|tag:count}
transProb = {}  # dict of {tag1==>tag2:prob}
stateCount = {} # dict of {tagName:count}
line = ''  # current line
tagStateList = []  # list of all tag state
tagStateDict = {} # dict of {idx:tagName}
wordsInModel = {}
outputContents = ''
totalTags = 0
marker = -1

""" UTILITY FUNCTIONS"""
### Function to read model file and import parameters
def read_and_process_model_file():
    # declare global variables
    global totalTags
    global marker
    global tagStateList
    global tagStateDict
    global wordsInModel
    global stateCount
    global transProb
    global emisProb

    # import model file
    modelFile = open(model_file, 'r', encoding='utf-8')
    lineCount = 0
    for line in modelFile:
        # Store number of tags in the tagset
        if lineCount == 0:
            lineCount += 1
            totalTags = int(line.split(':')[1])
            continue

        # Store list of tags
        if lineCount == 1:
            lineCount += 1
            # extract all tag in tagset
            tagSet = line.split('=*>')[1]
            # print(tagSet)

            # remove junk in the tagset
            tagSet = tagSet.strip('\n')
            tagSet = tagSet.split(',')
            tagSet.remove('')

            # extract list of all possible states/tags
            tagStateList = tagSet
            # print(tagStateList)
            continue

        # separate block of data
        if line == "Emission Probability:\n":
            marker = 0
            continue

        if line == "Transition Probability:\n":
            marker = 1
            continue

        if line == "State Count:\n":
            marker = 2
            continue

        # emission prob
        if marker == 0:
            # Sample emission prob: P(LONDRA|SP) =*> 0.00022
            # split each data block by =*>
            data = line.split('=*>')
            var1 = data[0] # extract P(Londra|SP)
            var1 = var1[2:len(var1) - 2] # extract Londra|SP
            corpusWord = var1.split('|')[0] # extract Londra
            wordsInModel[corpusWord] = 1  # update wordsInModel

            # get emission prob of dict: {word|tag:prob}
            emisProb[var1] = "{0:.5f}".format(float(data[1].strip('\n')))

        # transition prob
        if marker == 1:
            # Sample transition prob: SP==>FS=*>0.18004
            data = line.split('=*>')
            var2 = data[0]  # extract SP==>FS
            var2 = var2.replace('START_STATE', 'q0')  # refine

            # get transition prob dict: {tag1=>tag2:prob}
            transProb[var2] = "{0:.5f}".format(float(data[1].strip('\n')))

        # state count
        if marker == 2:
            # sample state count: SP=*>13657
            var3 = line.split('=*>')

            # get state count dict: {tag:count}
            stateCount[var3[0]] = int(var3[1].strip('\n'))


### Viterbi Algorithm for Decoding
# input is line
def viterbi_algorithm(line):
    global totalTags
    global marker
    global tagStateList
    global tagStateDict
    global wordsInModel
    global stateCount
    global transProb
    global emisProb
    global outputContents  # append to existing outputContents after each run of viterbi

    # initiate score
    score = 0

    # get word sequence from line
    wordSequence = line.split(' ')
    # print(wordSequence)

    # get length of the word sequence
    T = len(wordSequence)

    # initiate viterbi and backtrack matrices
    viterbi = [[0 for x in range(T)] for y in range(totalTags + 1)]  # numWords * numTag
    backtrack = [[0 for x in range(T)] for y in range(totalTags + 1)]

    # initialization of beginning of sentence
    for item1 in tagStateDict.keys():
        # get emission key = word|tag
        emissionKey = wordSequence[0] + '|' + tagStateDict[item1]

        # if a new word appears in the development corpus, set probability to 1
        if wordSequence[0] not in wordsInModel.keys():
            probabilityOfEmission = 1.0

        # if a new word|tag appears in the development corpus, set probability to 0
        elif emissionKey not in emisProb.keys():
            probabilityOfEmission = 0.0

        # else set data same as model
        else:
            probabilityOfEmission = float(emisProb[emissionKey])

        # set transition key of the beginning of sentence: q0-tagName (follow model format)
        transitionKey = 'q0-' + tagStateDict[item1]

        # if transition key not in model transition prob's keys, set it = 1/(numTags+numTagCount of 'q0')
        if transitionKey not in transProb.keys():
            probabilityOfTransition = float(1 / (int(stateCount['q0']) + totalTags))
        # else, set it equal to the model
        else:
            probabilityOfTransition = float(transProb[transitionKey])

        # update viterbi matrix: first element of each row = prior prob*emission
        viterbi[item1][0] = probabilityOfTransition * probabilityOfEmission

        # update backtrack matrix: pointing to q0
        backtrack[item1][0] = 0

    # For the next states do
    # loop through each position from 2nd word to end of sentence
    for item2 in range(1, T):
        # loop through end state tag2
        for toState in tagStateDict.keys():
            # loop through begin state tag1
            for fromState in tagStateDict.keys():
                # construct emission key = word|tag2
                emissionKey = wordSequence[item2] + '|' + tagStateDict[toState]

                if wordSequence[item2] not in wordsInModel.keys():
                    probabilityOfEmission = 1.0
                elif emissionKey not in emisProb.keys():
                    probabilityOfEmission = 0.0
                else:
                    probabilityOfEmission = float(emisProb[emissionKey])

                # set transition key of the beginning of sentence: tag1-tag2 (follow model format)
                transitionKey = tagStateDict[fromState] + '==>' + tagStateDict[toState]

                # if transition key not in model transition prob's keys, set it = 1/(numTags+numTagCount of 'q0')
                if transitionKey not in transProb.keys():
                    tagName = tagStateDict[fromState]
                    tagCount = stateCount[tagName]
                    probabilityOfTransition = float(1 / (int(tagCount) + totalTags))
                else:
                    probabilityOfTransition = float(transProb[transitionKey])

                # update score = previous score * transition prob * emission prob (see part C hw2 question 3)
                score = float(viterbi[fromState][item2 - 1]) * probabilityOfTransition * probabilityOfEmission

                # compare score to existing score at toState at word location to update backtrack pointer
                if score > float(viterbi[toState][item2]):
                    viterbi[toState][item2] = score
                    backtrack[toState][item2] = fromState
                else:
                    continue
    
    # obtain answer by evaluating the two matrices backwards
    best = 0  # initialize best index
    # loop through each tag
    for item3 in tagStateDict.keys():
        # compare score of prob corresponding to each index to the current best index
        if viterbi[item3][T - 1] > viterbi[best][T - 1]:
            # update index with most likely state
            best = item3

    # save answer of word/bestTag
    output_line = wordSequence[T - 1] + '/' + tagStateDict[best] + ' '

    # continuously update answer when backtracking
    for item4 in range(T - 1, 0, -1):
        best = backtrack[best][item4]
        # append answer to the left of previous answer to maintain order
        output_line = wordSequence[item4 - 1] + '/' + tagStateDict[best] + ' ' + output_line

    # save answers for each line
    outputContents += output_line + '\n'
    return outputContents

""" MAIN: run command: python hmmdecode3.py it_isdt_dev_raw.txt"""
def main():
    # globalize variables
    global tagStateList
    global tagStateDict
    global outputContents
    
    # import model file
    read_and_process_model_file()
    
    # loop through each tag to create values for tagStateDict
    i = 0
    for tagName in tagStateList:
        tagStateDict[i] = tagName
        i += 1
    
    # import raw development corpus    
    inputRawFileLoc = sys.argv[1]
    
    # open development file to read
    inputFile = open(inputRawFileLoc, 'r', encoding='utf-8')
    
    # open output file to write    
    outputFile = open(output_file, 'w', encoding="utf8")
    
    # loop through each line in inputFile to perform viterbi algorithm
    for line in inputFile:
        # pre-process each line before running viterbi
        outputContents = viterbi_algorithm(line.strip())
    
    # write results
    outputFile.write(outputContents)
    
    # Close files
    inputFile.close()
    outputFile.close()


if __name__ == "__main__":
    main()
