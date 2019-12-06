""" Import necessary library"""
from __future__ import division
import math # math
import pickle
import re  # regular expression operations
import sys  #  System-specific parameters and functions
import os
import fnmatch


""" Stop word list: import from Internet"""
# stopWords = ["'ll", "'tis", "'twas", "'ve", 'a', "a's", 'able', 'ableabout', 'about', 'above', 'abroad', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'ad', 'added', 'adj', 'adopted', 'ae', 'af', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'ag', 'again', 'against', 'ago', 'ah', 'ahead', 'ai', "ain't", 'aint', 'al', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ao', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'aq', 'ar', 'are', 'area', 'areas', 'aren', "aren't", 'arent', 'arise', 'around', 'arpa', 'as', 'aside', 'ask', 'asked', 'asking', 'asks', 'associated', 'at', 'au', 'auth', 'available', 'aw', 'away', 'awfully', 'az', 'b', 'ba', 'back', 'backed', 'backing', 'backs', 'backward', 'backwards', 'bb', 'bd', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'beings', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'bf', 'bg', 'bh', 'bi', 'big', 'bill', 'billion', 'biol', 'bj', 'bm', 'bn', 'bo', 'both', 'bottom', 'br', 'brief', 'briefly', 'bs', 'bt', 'but', 'buy', 'bv', 'bw', 'by', 'bz', 'c', "c'mon", "c's", 'ca', 'call', 'came', 'can', "can't", 'cannot', 'cant', 'caption', 'case', 'cases', 'cause', 'causes', 'cc', 'cd', 'certain', 'certainly', 'cf', 'cg', 'ch', 'changes', 'ci', 'ck', 'cl', 'clear', 'clearly', 'click', 'cm', 'cmon', 'cn', 'co', 'co.', 'com', 'come', 'comes', 'computer', 'con', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'copy', 'corresponding', 'could', "could've", 'couldn', "couldn't", 'couldnt', 'course', 'cr', 'cry', 'cs', 'cu', 'currently', 'cv', 'cx', 'cy', 'cz', 'd', 'dare', "daren't", 'darent', 'date', 'de', 'dear', 'definitely', 'describe', 'described', 'despite', 'detail', 'did', 'didn', "didn't", 'didnt', 'differ', 'different', 'differently', 'directly', 'dj', 'dk', 'dm', 'do', 'does', 'doesn', "doesn't", 'doesnt', 'doing', 'don', "don't", 'done', 'dont', 'doubtful', 'down', 'downed', 'downing', 'downs', 'downwards', 'due', 'during', 'dz', 'e', 'each', 'early', 'ec', 'ed', 'edu', 'ee', 'effect', 'eg', 'eh', 'eight', 'eighty', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'end', 'ended', 'ending', 'ends', 'enough', 'entirely', 'er', 'es', 'especially', 'et', 'et-al', 'etc', 'even', 'evenly', 'ever', 'evermore', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'fairly', 'far', 'farther', 'felt', 'few', 'fewer', 'ff', 'fi', 'fifteen', 'fifth', 'fifty', 'fify', 'fill', 'find', 'finds', 'fire', 'first', 'five', 'fix', 'fj', 'fk', 'fm', 'fo', 'followed', 'following', 'follows', 'for', 'forever', 'former', 'formerly', 'forth', 'forty', 'forward', 'found', 'four', 'fr', 'free', 'from', 'front', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthermore', 'furthers', 'fx', 'g', 'ga', 'gave', 'gb', 'gd', 'ge', 'general', 'generally', 'get', 'gets', 'getting', 'gf', 'gg', 'gh', 'gi', 'give', 'given', 'gives', 'giving', 'gl', 'gm', 'gmt', 'gn', 'go', 'goes', 'going', 'gone', 'good', 'goods', 'got', 'gotten', 'gov', 'gp', 'gq', 'gr', 'great', 'greater', 'greatest', 'greetings', 'group', 'grouped', 'grouping', 'groups', 'gs', 'gt', 'gu', 'gw', 'gy', 'h', 'had', "hadn't", 'hadnt', 'half', 'happens', 'hardly', 'has', 'hasn', "hasn't", 'hasnt', 'have', 'haven', "haven't", 'havent', 'having', 'he', "he'd", "he'll", "he's", 'hed', 'hell', 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'herse\xe2\x80\x9d', 'hes', 'hi', 'hid', 'high', 'higher', 'highest', 'him', 'himself', 'himse\xe2\x80\x9d', 'his', 'hither', 'hk', 'hm', 'hn', 'home', 'homepage', 'hopefully', 'how', "how'd", "how'll", "how's", 'howbeit', 'however', 'hr', 'ht', 'htm', 'html', 'http', 'hu', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'i.e.', 'id', 'ie', 'if', 'ignored', 'ii', 'il', 'ill', 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'inc.', 'indeed', 'index', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'inside', 'insofar', 'instead', 'int', 'interest', 'interested', 'interesting', 'interests', 'into', 'invention', 'inward', 'io', 'iq', 'ir', 'is', 'isn', "isn't", 'isnt', 'it', "it'd", "it'll", "it's", 'itd', 'itll', 'its', 'itself', 'itse\xe2\x80\x9d', 'ive', 'j', 'je', 'jm', 'jo', 'join', 'jp', 'just', 'k', 'ke', 'keep', 'keeps', 'kept', 'keys', 'kg', 'kh', 'ki', 'kind', 'km', 'kn', 'knew', 'know', 'known', 'knows', 'kp', 'kr', 'kw', 'ky', 'kz', 'l', 'la', 'large', 'largely', 'last', 'lately', 'later', 'latest', 'latter', 'latterly', 'lb', 'lc', 'least', 'length', 'less', 'lest', 'let', "let's", 'lets', 'li', 'like', 'liked', 'likely', 'likewise', 'line', 'little', 'lk', 'll', 'long', 'longer', 'longest', 'look', 'looking', 'looks', 'low', 'lower', 'lr', 'ls', 'lt', 'ltd', 'lu', 'lv', 'ly', 'm', 'ma', 'made', 'mainly', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', "mayn't", 'maynt', 'mc', 'md', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'member', 'members', 'men', 'merely', 'mg', 'mh', 'microsoft', 'might', "might've", "mightn't", 'mightnt', 'mil', 'mill', 'million', 'mine', 'minus', 'miss', 'mk', 'ml', 'mm', 'mn', 'mo', 'more', 'moreover', 'most', 'mostly', 'move', 'mp', 'mq', 'mr', 'mrs', 'ms', 'msie', 'mt', 'mu', 'much', 'mug', 'must', "must've", "mustn't", 'mustnt', 'mv', 'mw', 'mx', 'my', 'myself', 'myse\xe2\x80\x9d', 'mz', 'n', 'na', 'name', 'namely', 'nay', 'nc', 'nd', 'ne', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needed', 'needing', "needn't", 'neednt', 'needs', 'neither', 'net', 'netscape', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nf', 'ng', 'ni', 'nine', 'ninety', 'nl', 'no', 'no-one', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'np', 'nr', 'nu', 'null', 'number', 'numbers', 'nz', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'older', 'oldest', 'om', 'omitted', 'on', 'once', 'one', "one's", 'ones', 'only', 'onto', 'open', 'opened', 'opening', 'opens', 'opposite', 'or', 'ord', 'order', 'ordered', 'ordering', 'orders', 'org', 'other', 'others', 'otherwise', 'ought', "oughtn't", 'oughtnt', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'pa', 'page', 'pages', 'part', 'parted', 'particular', 'particularly', 'parting', 'parts', 'past', 'pe', 'per', 'perhaps', 'pf', 'pg', 'ph', 'pk', 'pl', 'place', 'placed', 'places', 'please', 'plus', 'pm', 'pmid', 'pn', 'point', 'pointed', 'pointing', 'points', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'pr', 'predominantly', 'present', 'presented', 'presenting', 'presents', 'presumably', 'previously', 'primarily', 'probably', 'problem', 'problems', 'promptly', 'proud', 'provided', 'provides', 'pt', 'put', 'puts', 'pw', 'py', 'q', 'qa', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'reserved', 'respectively', 'resulted', 'resulting', 'results', 'right', 'ring', 'ro', 'room', 'rooms', 'round', 'ru', 'run', 'rw', 's', 'sa', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sb', 'sc', 'sd', 'se', 'sec', 'second', 'secondly', 'seconds', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'sees', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'seventy', 'several', 'sg', 'sh', 'shall', "shan't", 'shant', 'she', "she'd", "she'll", "she's", 'shed', 'shell', 'shes', 'should', "should've", 'shouldn', "shouldn't", 'shouldnt', 'show', 'showed', 'showing', 'shown', 'showns', 'shows', 'si', 'side', 'sides', 'significant', 'significantly', 'similar', 'similarly', 'since', 'sincere', 'site', 'six', 'sixty', 'sj', 'sk', 'sl', 'slightly', 'sm', 'small', 'smaller', 'smallest', 'sn', 'so', 'some', 'somebody', 'someday', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'sr', 'st', 'state', 'states', 'still', 'stop', 'strongly', 'su', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure', 'sv', 'sy', 'system', 'sz', 't', "t's", 'take', 'taken', 'taking', 'tc', 'td', 'tell', 'ten', 'tends', 'test', 'text', 'tf', 'tg', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thatll', 'thats', 'thatve', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'therell', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'thereve', 'these', 'they', "they'd", "they'll", "they're", "they've", 'theyd', 'theyll', 'theyre', 'theyve', 'thick', 'thin', 'thing', 'things', 'think', 'thinks', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thought', 'thoughts', 'thousand', 'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'till', 'tip', 'tis', 'tj', 'tk', 'tm', 'tn', 'to', 'today', 'together', 'too', 'took', 'top', 'toward', 'towards', 'tp', 'tr', 'tried', 'tries', 'trillion', 'truly', 'try', 'trying', 'ts', 'tt', 'turn', 'turned', 'turning', 'turns', 'tv', 'tw', 'twas', 'twelve', 'twenty', 'twice', 'two', 'tz', 'u', 'ua', 'ug', 'uk', 'um', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'upwards', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'uucp', 'uy', 'uz', 'v', 'va', 'value', 'various', 'vc', 've', 'versus', 'very', 'vg', 'vi', 'via', 'viz', 'vn', 'vol', 'vols', 'vs', 'vu', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'wasn', "wasn't", 'wasnt', 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'web', 'webpage', 'website', 'wed', 'welcome', 'well', 'wells', 'went', 'were', 'weren', "weren't", 'werent', 'weve', 'wf', 'what', "what'd", "what'll", "what's", "what've", 'whatever', 'whatll', 'whats', 'whatve', 'when', "when'd", "when'll", "when's", 'whence', 'whenever', 'where', "where'd", "where'll", "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whim', 'whither', 'who', "who'd", "who'll", "who's", 'whod', 'whoever', 'whole', 'wholl', 'whom', 'whomever', 'whos', 'whose', 'why', "why'd", "why'll", "why's", 'widely', 'width', 'will', 'willing', 'wish', 'with', 'within', 'without', 'won', "won't", 'wonder', 'wont', 'words', 'work', 'worked', 'working', 'works', 'world', 'would', "would've", 'wouldn', "wouldn't", 'wouldnt', 'ws', 'www', 'x', 'y', 'ye', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'youll', 'young', 'younger', 'youngest', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve', 'yt', 'yu', 'z', 'za', 'zero', 'zm', 'zr']

# stopWords = open('stopWords.txt', 'r').read().splitlines()
# print(stopWords)
stopWords = ["each","has", "had", "having", "do", "does", "did", "doing", "few", "more", "most", "other", "some", "such", "no","about", "against", "between", "into", "through", "during", "before","i", "me", "my", "myself", "we", "our", "ours","and", "but", "if",  "so", "than", "too", "very", "s", "t", "can", "will", "just", "or", "because", "as", "until", "while", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",  "of", "at", "by", "for", "with",  "after", "above","her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "have",  "a", "an", "the",  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",  "nor",  "am", "is", "are", "was", "were", "be", "been", "being","not", "only", "own", "same","don", "should", "now"]

""" Utility function"""
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


"""Specify classes in the corpus"""
POSITIVE = 'positive'
NEGATIVE = 'negative'
DECEPTIVE = 'deceptive'
TRUTHFUL = 'truthful'
# classes = ["pos", "neg", "dec", "tru"]
classes = [POSITIVE, NEGATIVE, DECEPTIVE, TRUTHFUL]

# if __name__ == "main":
def main(input_path):
    """ Data path"""
    # print(sys.argv)
    # define the base path
    # input_path = os.path.dirname(os.path.realpath(__file__)) + "\op_spam_training_data\\"
    # input_path = str(sys.argv[0]) # for vocareum
    # print(str(sys.argv[0]) , "\n", str(sys.argv[1]), "\n", str(sys.argv[2]))
    print("Root directory of the training data: ", input_path)

    # #paths ot the training data
    # # positive deceptive
    # pos_dec = input_path+"positive_polarity\deceptive_from_MTurk\\"
    # # print(pos_dec)
    #
    # # positive truthful
    # pos_tru = input_path+"positive_polarity\\truthful_from_TripAdvisor\\"
    #
    # # negative deceptive
    # neg_dec = input_path +"negative_polarity\deceptive_from_MTurk\\"
    #
    # # negative truthful
    # neg_tru = input_path+"negative_polarity\\truthful_from_Web\\"

    # #paths ot the training data
    # # positive deceptive
    # pos_dec = r"."+input_path+"/positive_polarity/deceptive_from_MTurk/"
    # # print(pos_dec)
    #
    # # positive truthful
    # pos_tru = r"."+input_path+"/positive_polarity/truthful_from_TripAdvisor/"
    #
    # # negative deceptive
    # neg_dec = r"."+input_path +"/negative_polarity/deceptive_from_MTurk/"
    #
    # # negative truthful
    # neg_tru = r"."+input_path+"/negative_polarity/truthful_from_Web/"

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

    # specify path directory for each class: i.e. positive = positive&truthful + positive&deceptive
    path_directory = { POSITIVE: [pos_dec, pos_tru],
                       NEGATIVE: [neg_dec, neg_tru],
                       DECEPTIVE: [pos_dec, neg_dec],
                       TRUTHFUL: [pos_tru, neg_tru]}
    # print("Path Directory of each class: ", path_directory)

    """ INITIATE COUNTS"""
    # initiate total count for each class for prior probability calculation
    classCount =      { POSITIVE: 0,
                        NEGATIVE: 0,
                        DECEPTIVE:0,
                        TRUTHFUL: 0 }

    # initiate dictionaries for each class {word:count} for conditional probability calculation of each token
    positive_dic = {}
    negative_dic = {}
    truthful_dic = {}
    deceptive_dic = {}

    # general vocabolary to be shared in projects
    vocabulary = {}
    totalDocs = 0

    # Loop through each class
    for key, value in path_directory.items():
        print("Building vocab for", key)

        # loop through each directory path belong to each class
        for path in value:
            print("Building vocab for ", key,"in path ", path)

            # loop through each file in the directory
            for root, directoryName, fileNames in os.walk(path):

                # filter out file that ends with .txt (file name pattern matching)
                for fileName in fnmatch.filter(fileNames, '*.txt'):
                    # reconstruct file name
                    fileName = os.path.join(root, fileName)
                    # print("currently reading: ", fileName)

                    # process the file to get the list of tokens
                    fileTokens = processFile(fileName)
                    # print(fileTokens)

                    # Add tokens to corresponding buckets
                    if(key == POSITIVE):
                        positive_dic = addTokensToDict(positive_dic, fileTokens)
                        classCount[POSITIVE] += 1
                        vocabulary = addTokensToDict(vocabulary, fileTokens)

                    elif(key == NEGATIVE):
                        negative_dic = addTokensToDict(negative_dic, fileTokens)
                        classCount[NEGATIVE] += 1
                        vocabulary = addTokensToDict(vocabulary, fileTokens)

                    elif(key == DECEPTIVE):
                        deceptive_dic = addTokensToDict(deceptive_dic, fileTokens)
                        classCount[DECEPTIVE] += 1
                        vocabulary = addTokensToDict(vocabulary, fileTokens)

                    elif(key == TRUTHFUL):
                        truthful_dic = addTokensToDict(truthful_dic, fileTokens)
                        classCount[TRUTHFUL] += 1
                        vocabulary = addTokensToDict(vocabulary, fileTokens)

                    totalDocs += 1

    # Total document: divided by 2 to account for duplication of 2 parses through dataset
    totalDocs /= 2

    """ PRINT QC"""
    print("# of documents: ", totalDocs)
    print("Class count: ", classCount)
    # print("vocabulary: ", vocabulary)
    # print("Positive review: ", positive_dic)
    # print("Negative review: ", negative_dic)
    # print("Deceptive review: ", deceptive_dic)
    # print("Truthful review: ", truthful_dic)

    """naive bayes model's parameter"""
    # Prior probability
    priorProb = {}

    # vocab size: total number of occurrences of all words
    vocabSize = getDictSize(vocabulary)

    # vocab len: total number of words in the vocabulary
    vocabLen = len(vocabulary)

    print("vocabLen: ", vocabLen)
    print("vocabSize: ", vocabSize)

    # initiate word counts (including outside words from vocab) for each class
    countPos = {}
    countNeg = {}
    countTru = {}
    countDec = {}

    # conditional probability
    condProb = {}
    condProbPos = {}
    condProbNeg = {}
    condProbTru = {}
    condProbDec = {}

    """ PERFORM CALCULATION"""
    # # Total document
    # totalDocs = 0
    # for category in classes:
    #     totalDocs += classCount[category]
    # print("# of documents: ", totalDocs/2)

    # Individual calculation
    for category in classes:
        # Prior probability of each class:
        priorProb[category] = classCount[category] / totalDocs

        # for each class
        if category == POSITIVE:
            # loop through each word
            for word in vocabulary:
                # if the word appears in the dict of training data for that class
                if word in positive_dic:
                    countPos[word] = positive_dic[word]
                # if not, set count to zero
                else:
                    countPos[word] = 0

            # get the size of the total count of that class
            wordCount1Class = getDictSize(countPos)
            # vocabLen = len(vocabulary)

            # loop again for all words
            for word in vocabulary:
                # calculate conditional probability of each word given this class
                # add one smoothing for each word in case its count = 0
                condProbPos[word] = (countPos[word] + 1) / (vocabLen + wordCount1Class)

        elif category == NEGATIVE:
            for word in vocabulary:
                if word in negative_dic:
                    countNeg[word] = negative_dic[word]
                else:
                    countNeg[word] = 0
            wordCount1Class = getDictSize(countNeg)
            vocabLen = len(vocabulary)
            for word in vocabulary:
                condProbNeg[word] = (countNeg[word] + 1) / (vocabLen + wordCount1Class)

        elif category == TRUTHFUL:
            for word in vocabulary:
                if word in truthful_dic:
                    countTru[word] = truthful_dic[word]
                else:
                    countTru[word] = 0
            wordCount1Class = getDictSize(countTru)
            vocabLen = len(vocabulary)
            for word in vocabulary:
                condProbTru[word] = (countTru[word] + 1) / (vocabLen + wordCount1Class)

        elif category == DECEPTIVE:
            for word in vocabulary:
                if word in deceptive_dic:
                    countDec[word] = deceptive_dic[word]
                else:
                    countDec[word] = 0
            wordCount1Class = getDictSize(countDec)
            vocabLen = len(vocabulary)
            for word in vocabulary:
                condProbDec[word] = (countDec[word] + 1) / (vocabLen + wordCount1Class)

    """ PRINT TO QC """
    # print("Prior probabilities: ", priorProb)



    """WRITE OUTPUT TO MODEL.TXT"""
    # open file to write
    fileOut = open('model.txt', 'w')

    # Info
    # fileOut.write("""Naive Bayes text classification:\n
    # 1) Specify path folder of training data, go to each text file\n
    # 2) Do tokenization: separate words, lowercase all words, \n
    # 3) Add words to corresponding baskets: pd,pt,nd,nt \n
    #  Based on training data's classification, you construct a dictionary with keys are words and values are another dict{category: countCategory} {word1:{pd:countPd, pt:countPt,...},word2:{},...}\n
    # 4) keep track of the count and calculate \n
    #  Naive bayes: calculate posterior based on prior and conditional probabilites \n
    # P(class|words) ~ P(words|class)*P(class) ~ P(word1|class)^k1*P(word2|class)^k2...*P(class)\n""")

    # write vocabulary
    fileOut.write("vocabulary " + str(vocabulary) + "\n")

    fileOut.write("Prior probability: " + str(priorProb)+ "\n")

    fileOut.write("Conditional probability of Deceptive words: " + str(condProbDec) + "\n")

    fileOut.write("Conditional probability of Truthful words: " + str(condProbTru)+ "\n")

    fileOut.write("Conditional probability of Negative words: " + str(condProbNeg)+ "\n")

    fileOut.write("Conditional probability of Positive words: " + str(condProbPos)+ "\n")

    # close the file
    fileOut.close()

    # write to nb model file
    model_file = "nbmodel.txt"
    fo = open(model_file, 'w')

    fo.write(str(vocabulary))
    fo.write("\n")

    fo.write(str(priorProb))
    fo.write("\n")

    fo.write(str(condProbDec))
    fo.write("\n")

    fo.write(str(condProbTru))
    fo.write("\n")

    fo.write(str(condProbNeg))
    fo.write("\n")

    fo.write(str(condProbPos))

    fo.close()

""" RUN THE PROGRAM"""
# For local path
# inPath = os.path.dirname(os.path.realpath(__file__)) + "\\op_spam_training_data\\"

# for command line invoke: python ./nblearn.py ./op_spam_training_data
inPath = str(sys.argv[-1])
print("inPath: ", inPath)
main(inPath)
    


