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
import time
import re

"""" DECLARE VARIABLES"""
model_file = "hmmmodel.txt"  # output file for hmm model
output_file = "hmmoutput.txt" # output file
emisProbDict = {}  # dict of {word|tag:count}
transProbDict = {}  # dict of {tag1=>tag2:prob}
tagCountDict = {} # dict of {tagName:count}
line = ''  # current line
tagList = []  # list of all tag state
tagDict = {} # dict of {idx:tagName}
allWordsFromModel = {}  # dict of all words from training corpus
outputContents = ''  # content to write to output
totalTags = 0  
lineIdx = -1  # index to mark line

""" UTILITY FUNCTIONS"""

# total size of the vocab: input is a dict (Not needed)
def getDictSize(vocab):
    output = 0
    for word in vocab:
        output += vocab[word]
    return output


# Function to remove punctuation from a token list [] (not needed)
def removePunctuations(tokenList):
    output = []
    # loop through each token
    for token in tokenList:
        # to do a regex substitution: remove all except letter, number and inherent white spaces
        token = re.sub('[^a-zA-Z0-9\n\-]', '', token)
        output.append(token)
    return output

### Viterbi Algorithm for Decoding
# input is line
def viterbi_decoder(line):
    global totalTags
    global lineIdx
    global tagList
    global tagDict
    global allWordsFromModel
    global tagCountDict
    global transProbDict
    global emisProbDict
    global outputContents  # append to existing outputContents after each run of viterbiMat

    # initiate cumProb
    cumProb = 0

    # get word sequence from line
    wordSeq = line.split(' ')
    # print(wordSeq)

    # get length of the word sequence
    wordSeqLen = len(wordSeq)

    # initiate viterbiMat and backtrackMat matrices
    viterbiMat = [[0 for x in range(wordSeqLen)] for y in range(totalTags + 1)]  # numWords * numTag
    backtrackMat = [[0 for x in range(wordSeqLen)] for y in range(totalTags + 1)]

    # initialization of beginning of sentence
    for idx1 in tagDict.keys():
        # get emission key = word|tag
        emis = wordSeq[0] + '|' + tagDict[idx1]

        # if a new word appears in the development corpus, set probability to 1 (ignore emission and use transitional probability alone)
        if wordSeq[0] not in allWordsFromModel.keys():
            emisProb = 1.0

        # if a new word|tag appears in the development corpus, set probability to 0
        elif emis not in emisProbDict.keys():
            emisProb = 0.0
            # continue

        # else set data same as model
        else:
            emisProb = float(emisProbDict[emis])
            # if emisProb == 0:
            #     continue

        # set transition key of the beginning of sentence: q0-tagName (follow model format)
        trans = 'q0-' + tagDict[idx1]

        # if transition key not in model transition prob's keys, set it = 1/(numTags+numTagCount of 'q0')
        if trans not in transProbDict.keys():
            transProb = float(1 / (int(tagCountDict['q0']) + totalTags))
        # else, set it equal to the model
        else:
            transProb = float(transProbDict[trans])
            # if transProb == 0:
            #     continue

        # update viterbiMat matrix: first element of each row = prior prob*emission
        viterbiMat[idx1][0] = transProb * emisProb

        # update backtrackMat matrix: pointing to q0
        backtrackMat[idx1][0] = 0

    # For the next states do
    # loop through each position from 2nd word to end of sentence
    for idx2 in range(1, wordSeqLen):
        # loop through end state tag2
        for endTag in tagDict.keys():
            # loop through begin state tag1
            for beginTag in tagDict.keys():
                # construct emission key = word|tag2
                emis = wordSeq[idx2] + '|' + tagDict[endTag]

                # unseen word never encountered in training data
                if wordSeq[idx2] not in allWordsFromModel.keys():
                    emisProb = 1.0
                elif emis not in emisProbDict.keys():
                    emisProb = 0.0
                    continue
                else:
                    emisProb = float(emisProbDict[emis])
                    # no need to proceed further as all subsequent prob = 0
                    if emisProb == 0:
                        continue

                # set transition key of the beginning of sentence: tag1-tag2 (follow model format)
                trans = tagDict[beginTag] + '=>' + tagDict[endTag]

                # if transition key not in model transition prob's keys, set it = 1/(numTags+numTagCount of 'q0')
                if trans not in transProbDict.keys():
                    tagName = tagDict[beginTag]
                    tagCount = tagCountDict[tagName]
                    transProb = float(1 / (int(tagCount) + totalTags))
                else:
                    transProb = float(transProbDict[trans])
                    if transProb == 0: # no need to proceed further as all subsequent prob = 0
                        continue

                # update cumProb = previous cumProb * transition prob * emission prob (see part C hw2 question 3)
                cumProb = float(viterbiMat[beginTag][idx2 - 1]) * transProb * emisProb
                if cumProb == 0:  # no need to proceed further as all subsequent prob = 0
                    continue

                # compare cumProb to existing cumProb at endTag at word location to update backtrackMat pointer
                if cumProb > float(viterbiMat[endTag][idx2]):
                    viterbiMat[endTag][idx2] = cumProb
                    backtrackMat[endTag][idx2] = beginTag
                else:
                    continue
    
    # obtain answer by evaluating the two matrices backwards
    bestIdx = 0  # initialize bestIdx index
    # loop through each tag
    for idx3 in tagDict.keys():
        # compare cumProb of prob corresponding to each index to the current bestIdx index
        if viterbiMat[idx3][wordSeqLen - 1] > viterbiMat[bestIdx][wordSeqLen - 1]:
            # update index with most likely state
            bestIdx = idx3

    # save answer of word/bestTag
    outputForEachLine = wordSeq[wordSeqLen - 1] + '/' + tagDict[bestIdx] + ' '

    # continuously update answer when backtracking
    for idx4 in range(wordSeqLen - 1, 0, -1):
        bestIdx = backtrackMat[bestIdx][idx4]
        # append answer to the left of previous answer to maintain order
        outputForEachLine = wordSeq[idx4 - 1] + '/' + tagDict[bestIdx] + ' ' + outputForEachLine

    # save answers for each line
    outputContents += outputForEachLine + '\n'
    return outputContents


### Function to read model file and import parameters
def model_file_reader():
    # declare global variables
    global totalTags
    global lineIdx
    global tagList
    global tagDict
    global allWordsFromModel
    global tagCountDict
    global transProbDict
    global emisProbDict

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
            tagSet = line.split('==>')[1]
            # print(tagSet)

            # remove junk in the tagset
            tagSet = tagSet.strip('\n')
            tagSet = tagSet.split(',')
            tagSet.remove('')

            # extract list of all possible states/tags
            tagList = tagSet
            # print(tagList)
            continue

        # separate block of data
        if line == "Emission Probability:\n":
            lineIdx = 0
            continue

        if line == "Transition Probability:\n":
            lineIdx = 1
            continue

        if line == "State Count:\n":
            lineIdx = 2
            continue

        # emission prob
        if lineIdx == 0:
            # Sample emission prob: P(LONDRA|SP) ==> 0.00022
            # split each data block by ==>
            data = line.split('==>')
            term1 = data[0] # extract P(Londra|SP)
            term1 = term1[2:len(term1) - 2] # extract Londra|SP
            trainWord = term1.split('|')[0] # extract Londra
            allWordsFromModel[trainWord] = 1  # update allWordsFromModel

            # get emission prob of dict: {word|tag:prob}
            emisProbDict[term1] = "{0:.5f}".format(float(data[1].strip('\n')))

        # transition prob
        if lineIdx == 1:
            # Sample transition prob: SP=>FS==>0.18004
            data = line.split('==>')
            term2 = data[0]  # extract SP=>FS
            # term2 = term2.replace('START_STATE', 'q0')  # refine

            # get transition prob dict: {tag1=>tag2:prob}
            transProbDict[term2] = "{0:.5f}".format(float(data[1].strip('\n')))

        # state count
        if lineIdx == 2:
            # sample state count: SP==>13657
            term3 = line.split('==>')

            # get state count dict: {tag:count}
            tagCountDict[term3[0]] = int(term3[1].strip('\n'))

""" MAIN: run command: python hmmdecode3.py it_isdt_dev_raw.txt"""
def main():
    # globalize variables
    global tagList
    global tagDict
    global outputContents
    
    # import model file
    model_file_reader()
    
    # loop through each tag to create values for tagDict
    i = 0
    for tagName in tagList:
        tagDict[i] = tagName
        i += 1
    
    # import raw development corpus    
    inputRawFileLoc = sys.argv[1]
    
    # open development file to read
    inputFile = open(inputRawFileLoc, 'r', encoding='utf-8')
    
    # open output file to write    
    outputFile = open(output_file, 'w', encoding="utf8")
    
    # loop through each line in inputFile to perform viterbiMat algorithm
    for line in inputFile:
        # pre-process each line before running viterbiMat
        outputContents = viterbi_decoder(line.strip())
    
    # write results
    outputFile.write(outputContents)
    
    # Close files
    inputFile.close()
    outputFile.close()


if __name__ == "__main__":
    start_time = time.time()

    main()

    # signal time elapsed
    # print("--- %s seconds ---" % (time.time() - start_time))