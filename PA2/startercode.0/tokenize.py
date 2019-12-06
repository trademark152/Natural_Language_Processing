stopWords = ["each","has", "had", "having", "do", "does", "did", "doing", "few", "more", "most", "other", "some", "such", "no","about", "against", "between", "into", "through", "during", "before","i", "me", "my", "myself", "we", "our", "ours","and", "but", "if",  "so", "than", "too", "very", "s", "t", "can", "will", "just", "or", "because", "as", "until", "while", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",  "of", "at", "by", "for", "with",  "after", "above","her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "have",  "a", "an", "the",  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",  "nor",  "am", "is", "are", "was", "were", "be", "been", "being","not", "only", "own", "same","don", "should", "now"]

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