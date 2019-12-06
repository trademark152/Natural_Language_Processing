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
import io

"""" DECLARE VARIABLES"""
model_file = "hmmmodel.txt"  # output file for hmm model
wordTagDict = []  # dict of {word|tag pair: occurrences}
beginTag = 'q0' # beginning of sentences tag
endTag = 'qE' # end of sentences tag
tagCountDict = {} # dict of {tag:count}
emisProb = {}  # dict of {word|tag:count}
tagTransCountDict = {} # dict of {tag1==>tag2:count}
stateCount = {} # dict of {tagName:count}
transProb = {}  # dict of {tag1==>tag2:prob}
currentTag = ''  # current tag
line = ''  # current line




""" UTILITY FUNCTIONS"""
### Function to count transitions from previous to next state (tag)
# input are preceding and current tag
# output: stateTransitionCount dict: {[beginTag ==> currentTag: count]}
def count_state_transitions(beginTag, currentTag):
    # create key in format: Tag1==>Tag2
    transition = beginTag + '==>' + currentTag

    # if this transition is already in dict
    if transition in tagTransCountDict:
        tagTransCountDict[transition] += 1
    else:
        tagTransCountDict[transition] = 1


### Function to count emissions from each tag: P(word|tag)
# Input are all separated words and line they belong
def count_emissions(words,line):
    # if words already in the emission dict:
    if words in emisProb.keys():
        emisProb[words] += line.count(words)
    # if not:
    else:
        emisProb[words] = line.count(words)

### Function to count total occurences of label to ANY label
def count_tag_occurrence(beginTag):
    if beginTag in stateCount.keys():
        stateCount[beginTag] += 1
    else:
        stateCount[beginTag] = 1
        # print(beginTag)

### Function to count the individual and total tags of each line = word1|tag1 word2|tag2... .|FS
def count_tags(line):
    # initiate start of sentence
    beginTag = 'q0'
    endTag = 'qE'
    currentTag = ''
    # print(line)

    # Tag counter by looping through each word|tag in
    for words in line:
        # print(words)
        # find index to separate word from tag
        i = 0
        # loop through each position of word|tag
        for pos in range(len(words)-1, 0, -1):
            # stop when the slash is met
            if words[pos] == '/':
                i = pos
                break

        # Use the index to separate the tag from '/' to end
        currentTag = words[i+1:]

        # update tag count
        if currentTag in tagCountDict:
            tagCountDict[currentTag] += 1
        else:
            tagCountDict[currentTag] = 1
            # print(currentTag)

        # Special case of first and last words??
        if beginTag == '' or currentTag == '':
            beginTag = currentTag
            continue

        # Special case
        # if currentTag == '_':
        #     beginTag = 'q0'
        #     # print("Special case: ", beginTag)
        #     continue

        # update transition tag1==>tag2 count
        count_state_transitions(beginTag, currentTag)

        # Count total word|tag
        count_emissions(words, line)

        # Count total tags
        count_tag_occurrence(beginTag)

        if currentTag == '_':
            count_tag_occurrence('_')

        # update preceding tag to current
        beginTag = currentTag


### Function to get the tags from the training corpus
def get_tags():
    for line in wordTagDict:
        count_tags(line)

    # print(tags)
    # return tags

### Function to read Input Training Corpus File and store all lines ###
# input: directory of training file
# output: all word|tag pairs
def read_training_corpus_file(train_file):
    text = open(train_file, 'r', encoding="utf8")
    # with io.open(train_file,'r', encoding="utf8") as file:
    #     text = file.read()
    #
    # with open(train_file) as file:
    #     text = file.readlines()
    # print(text)
    for line in text:
        wordTagDict.append(line.split())

    text.close()


### Function to calculate emission probabilitiesand write to the model file ##
# input: file to write
def calculate_and_write_emission_probability(file):
    # loop through keys of emisProb i.e. word|tag
    for item in emisProb.keys():
        # Find index to separate tags
        i = 0
        for pos in range(len(item)-1, 0, -1):
            if item[pos] == '/':
                i = pos
                break

        # obtain tag and word (check if slash is excluded)
        tag = item[i+1:]
        word = item[0:i]

        # if count of that tag is non-zero
        if tagCountDict[tag] > 0:
            # calculate emission probability = # of word|tag/# of tag
            emisProb[item] = emisProb[item] / tagCountDict[tag]

            # QC if probability is greater than 1??
            if emisProb[item] > 1:
                emisProb[item] = 1

            # save emission prob answers as long formatted decimal
            var1 = str("{0:.5f}".format(emisProb[item]))

            # format emission prob answer: P(word|tag) =*> emission Prob
            output = 'P('+word+'|'+tag+') =*> ' + var1 + '\n'

            # write to file
            file.write(output)

### Function to calculate transition probabilities with smoothing and write to the model file
# input: file to write
def calculate_and_write_transition_probability(file):
    # loop through keys of state transition, i.e. tag1==>tag2
    for item in tagTransCountDict.keys():
        # extract starting tag
        startTag = item.split('==>')[0]

        # extract ending tag
        nextTag = item.split('==>')[1]
        #print(startTag + "->" + nextTag)

        # if the count of the start tag is non zero in the corpus
        if stateCount[startTag] > 0:
            # calculate transition probability with smoothing + 1
            transProb[item] = (tagTransCountDict[item] + 1) / (stateCount[startTag] + len(tagCountDict))

            # if starting tag is q0, i.e. beginning of sentence
            if startTag == 'q0':
                # save transition prob as start_state
                var1 = str("{0:.5f}".format(transProb[item]))
                output = 'START_STATE-' + nextTag + '=*>' + var1 + '\n'
            else:
                var2 = str("{0:.5f}".format(transProb[item]))
                output = item + '=*>' + var2 + '\n'

            # write to file
            file.write(output)

            # reset output
            output = ''


### Function to write to a model file ###
def write_to_model_file():
    # Open model file to write
    file = open(model_file, 'w+',encoding="utf8")

    # write number of tag in the tagset
    file.write('No. of tags:' + str(len(tagCountDict)) + '\n')

    # save all tags in tagset
    tags = []  # all tags
    for tag in tagCountDict.keys():
        # tags.append(str(tag+','))
        tags.append(tag)
    # print(tags)
    file.write('Tags=*>')
    for tag in tags:
        file.write(tag+',')
    file.write('\n')

    # write count of each tag
    file.write('State Count:\n')
    # print(stateCount.keys())
    for item in stateCount.keys():
        file.write(item + '=*>' + str(stateCount[item]) + '\n')

    # write transition probability
    file.write('Transition Probability:\n')
    calculate_and_write_transition_probability(file)

    #  write emission probability
    file.write('Emission Probability:\n')
    calculate_and_write_emission_probability(file)

    # close file
    file.close()



if __name__=="__main__":
    # get training corpus from command line: hmmlearn3.py it_isdt_train_tagged.txt
    train_file = sys.argv[1]
    # print(train_file)

    # read training corpus
    read_training_corpus_file(train_file)

    # get all tags
    tags = get_tags()

    # open model file to write
    open(model_file, "w")

    # write to model file
    write_to_model_file()

    print(tagCountDict)
    print(stateCount)