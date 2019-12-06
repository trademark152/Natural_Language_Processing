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
  - trans probabilities are the conditional probabilities of tags sequence
  - emission probabilities are the conditional probabilities of words given tags. 
  - The start state is the beginning of a sentence, which is not a partof-speech tag
"""

""" IMPORT LIBRARIES"""
import sys
import time
import re

"""" DECLARE VARIABLES"""
model_file = "hmmmodel.txt"  # output file for hmm model
tagTransCountDict = {} # dict of {tag1=>tag2:count}
beginTag = 'q0' # beginning of sentences tag
endTag = 'qE' # end of sentences tag
tagCountDict = {} # dict of {tag:count}
emisProb = {}  # dict of {word|tag:count}
currentTag = ''  # current tag
line = ''  # current line
transProb = {}  # dict of {tag1=>tag2:prob}
wordTagDict = []  # dict of {word|tag pair: occurrences}

""" CLASS """
stopWords = []

# create a document class: (might need fixing)
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



""" UTILITY FUNCTIONS"""
### Function to count transitions from previous to next state (tag)
# input are preceding and current tag
# output: stateTransitionCount dict: {[beginTag => currentTag: count]}
def tag_trans_counter(beginTag, currentTag):
    # create key in format: Tag1=>Tag2
    trans = beginTag + '=>' + currentTag

    # if this trans is already in dict
    if trans in tagTransCountDict:
        tagTransCountDict[trans] += 1
    else:
        tagTransCountDict[trans] = 1


### Function to count emissions from each tag: P(word|tag)
# Input are all separated words and line they belong
def tag_emis_counter(wordTag, line):
    # if wordTag already in the emission dict:
    if wordTag in emisProb.keys():
        emisProb[wordTag] += line.count(wordTag)
    # if not:
    else:
        emisProb[wordTag] = line.count(wordTag)

### Function to count total occurences of label to ANY label
def tag_total_counter(beginTag):
    if beginTag in tagCountDict.keys():
        tagCountDict[beginTag] += 1
    else:
        tagCountDict[beginTag] = 1
        # print(beginTag)

# remove all punctuations from a text (not needed)
def removePuncs(text):
    # to do a regex substitution: remove all except letter, number and inherent white spaces
    return re.sub(r'[^a-zA-Z0-9 ]', r'', text)

# remove words from stopWords (not needed)
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

### Function to count the individual and total tags of each line = word1|tag1 word2|tag2... .|FS
def tag_counter(line):
    # initiate start of sentence
    beginTag = 'q0'
    currentTag = ''
    # print(line)

    # Tag counter by looping through each word|tag in
    for wordTag in line:
        # print(wordTag)
        # find index to separate word from tag
        idx = 0
        # loop through each position of word|tag
        for pos in range(len(wordTag)-1, 0, -1):
            # stop when the slash is met
            if wordTag[pos] == '/':
                idx = pos
                break

        # Use the index to separate the tag from '/' to end
        currentTag = wordTag[idx+1:]

        # update tag count
        if currentTag in tagCountDict:
            tagCountDict[currentTag] += 1
        else:
            tagCountDict[currentTag] = 1
            # print(currentTag)

        # Special case of first and last wordTag??
        if beginTag == '' or currentTag == '':
            beginTag = currentTag
            continue

        # Special case
        # if currentTag == '_':
        #     beginTag = 'q0'
        #     # print("Special case: ", beginTag)
        #     continue

        # update trans tag1=>tag2 count
        tag_trans_counter(beginTag, currentTag)

        # Count total word|tag
        tag_emis_counter(wordTag, line)

        # Count total tags
        tag_total_counter(beginTag)

        if currentTag == '_':
            tag_total_counter('_')

        # update preceding tag to current
        beginTag = currentTag


### Function to get the tags from the training corpus
def all_tag_getter():
    for line in wordTagDict:
        tag_counter(line)

    # print(tags)
    # return tags

### Function to read Input Training Corpus File and store all lines ###
# input: directory of training file
# output: all word|tag pairs
def train_file_reader(train_file):
    text = open(train_file, 'r', encoding="utf8")
    for line in text:
        wordTagDict.append(line.split())

    text.close()


### Function to calculate emission probabilitiesand write to the model file ##
# input: file to write
def emis_prob_calculator(file):
    # loop through keys of emisProb idx.e. word|tag
    for wordTagPair in emisProb.keys():
        # Find index to separate tags
        idx = 0
        for pos in range(len(wordTagPair)-1, 0, -1):
            if wordTagPair[pos] == '/':
                idx = pos
                break

        # obtain tag and word (check if slash is excluded)
        tag = wordTagPair[idx+1:]
        word = wordTagPair[0:idx]

        # if count of that tag is non-zero
        if tagCountDict[tag] > 0:
            # calculate emission probability = # of word|tag/# of tag
            emisProb[wordTagPair] = emisProb[wordTagPair] / tagCountDict[tag]

            # QC if probability is greater than 1??
            if emisProb[wordTagPair] > 1:
                emisProb[wordTagPair] = 1

            # save emission prob answers as long formatted decimal
            var1 = str("{0:.5f}".format(emisProb[wordTagPair]))

            # format emission prob answer: P(word|tag) ==> emission Prob
            output = 'P('+word+'|'+tag+') ==> ' + var1 + '\n'

            # write to file
            file.write(output)

### Function to calculate trans probabilities with smoothing and write to the model file
# input: file to write
def trans_prob_calculator(file):
    # loop through keys of state trans, idx.e. tag1=>tag2
    for wordTagPair in tagTransCountDict.keys():
        # extract starting tag
        startTag = wordTagPair.split('=>')[0]

        # extract ending tag
        nextTag = wordTagPair.split('=>')[1]
        #print(startTag + "->" + nextTag)

        # if the count of the start tag is non zero in the corpus
        if tagCountDict[startTag] > 0:
            # calculate trans probability with smoothing + 1
            transProb[wordTagPair] = (tagTransCountDict[wordTagPair] + 1) / (tagCountDict[startTag] + len(tagCountDict))

            # if starting tag is q0, idx.e. beginning of sentence
            if startTag == 'q0':
                # save trans prob as start_state
                var1 = str("{0:.5f}".format(transProb[wordTagPair]))
                output = 'q0-' + nextTag + '==>' + var1 + '\n'
            else:
                var2 = str("{0:.5f}".format(transProb[wordTagPair]))
                output = wordTagPair + '==>' + var2 + '\n'

            # write to file
            file.write(output)

            # reset output
            output = ''

def word_replacer(s, old, new, occurrence):
    newVersion = s.rsplit(old, occurrence)
    return new.join(newVersion)

def word_cleaner(word):
    cleanedVersion = word.replace('/', '_')
    cleanedVersion = word_replacer(cleanedVersion, '_', '/', 1)
    return cleanedVersion

### Function to write to a model file ###
def write_to_model_file():
    # Open model file to write
    file = open(model_file, 'w+',encoding="utf8")

    # write number of tag in the tagset
    file.write('No. of tags:' + str(len(tagCountDict)) + '\n')

    # save all tags in tagset
    allTags = []  # all tags
    for tag in tagCountDict.keys():
        # tags.append(str(tag+','))
        allTags.append(tag)
    # print(tags)
    file.write('Tags==>')
    for tag in allTags:
        file.write(tag+',')
    file.write('\n')

    # write count of each tag
    file.write('State Count:\n')
    # print(tagCountDict.keys())
    for wordTagPair in tagCountDict.keys():
        file.write(wordTagPair + '==>' + str(tagCountDict[wordTagPair]) + '\n')

    # write trans probability
    file.write('Transition Probability:\n')
    trans_prob_calculator(file)

    #  write emission probability
    file.write('Emission Probability:\n')
    emis_prob_calculator(file)

    # close file
    file.close()



if __name__=="__main__":
    # get training corpus from command line: hmmlearn3.py it_isdt_train_tagged.txt
    train_file = sys.argv[1]
    # print(train_file)

    # start timer
    start_time = time.time()

    # read training corpus
    train_file_reader(train_file)

    # get all tags
    all_tag_getter()

    # open model file to write
    open(model_file, "w")

    # write to model file
    write_to_model_file()

    # signal time elapsed
    # print("--- %s seconds ---" % (time.time() - start_time))
