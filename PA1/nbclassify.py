#!/usr/bin/python
from __future__ import division
import os
import fnmatch
import re
import sys
import math
import glob


""" Import data from nbmode.txt"""
# open file
model_file = "nbmodel.txt"
fileIn = open(model_file, "r")

# import data
vocabulary = eval(fileIn.readline())
# print(vocabulary)
priorProb = eval(fileIn.readline())
# print(priorProb)
condProbDec = eval(fileIn.readline())
# print(condProbDec)
condProbTru = eval(fileIn.readline())
condProbNeg = eval(fileIn.readline())
condProbPos = eval(fileIn.readline())

# close file
fileIn.close()

# stop Words:
# stopWords = ["'ll", "'tis", "'twas", "'ve", 'a', "a's", 'able', 'ableabout', 'about', 'above', 'abroad', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'ad', 'added', 'adj', 'adopted', 'ae', 'af', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'ag', 'again', 'against', 'ago', 'ah', 'ahead', 'ai', "ain't", 'aint', 'al', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'ao', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'aq', 'ar', 'are', 'area', 'areas', 'aren', "aren't", 'arent', 'arise', 'around', 'arpa', 'as', 'aside', 'ask', 'asked', 'asking', 'asks', 'associated', 'at', 'au', 'auth', 'available', 'aw', 'away', 'awfully', 'az', 'b', 'ba', 'back', 'backed', 'backing', 'backs', 'backward', 'backwards', 'bb', 'bd', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'beings', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'bf', 'bg', 'bh', 'bi', 'big', 'bill', 'billion', 'biol', 'bj', 'bm', 'bn', 'bo', 'both', 'bottom', 'br', 'brief', 'briefly', 'bs', 'bt', 'but', 'buy', 'bv', 'bw', 'by', 'bz', 'c', "c'mon", "c's", 'ca', 'call', 'came', 'can', "can't", 'cannot', 'cant', 'caption', 'case', 'cases', 'cause', 'causes', 'cc', 'cd', 'certain', 'certainly', 'cf', 'cg', 'ch', 'changes', 'ci', 'ck', 'cl', 'clear', 'clearly', 'click', 'cm', 'cmon', 'cn', 'co', 'co.', 'com', 'come', 'comes', 'computer', 'con', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'copy', 'corresponding', 'could', "could've", 'couldn', "couldn't", 'couldnt', 'course', 'cr', 'cry', 'cs', 'cu', 'currently', 'cv', 'cx', 'cy', 'cz', 'd', 'dare', "daren't", 'darent', 'date', 'de', 'dear', 'definitely', 'describe', 'described', 'despite', 'detail', 'did', 'didn', "didn't", 'didnt', 'differ', 'different', 'differently', 'directly', 'dj', 'dk', 'dm', 'do', 'does', 'doesn', "doesn't", 'doesnt', 'doing', 'don', "don't", 'done', 'dont', 'doubtful', 'down', 'downed', 'downing', 'downs', 'downwards', 'due', 'during', 'dz', 'e', 'each', 'early', 'ec', 'ed', 'edu', 'ee', 'effect', 'eg', 'eh', 'eight', 'eighty', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'end', 'ended', 'ending', 'ends', 'enough', 'entirely', 'er', 'es', 'especially', 'et', 'et-al', 'etc', 'even', 'evenly', 'ever', 'evermore', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'fairly', 'far', 'farther', 'felt', 'few', 'fewer', 'ff', 'fi', 'fifteen', 'fifth', 'fifty', 'fify', 'fill', 'find', 'finds', 'fire', 'first', 'five', 'fix', 'fj', 'fk', 'fm', 'fo', 'followed', 'following', 'follows', 'for', 'forever', 'former', 'formerly', 'forth', 'forty', 'forward', 'found', 'four', 'fr', 'free', 'from', 'front', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthermore', 'furthers', 'fx', 'g', 'ga', 'gave', 'gb', 'gd', 'ge', 'general', 'generally', 'get', 'gets', 'getting', 'gf', 'gg', 'gh', 'gi', 'give', 'given', 'gives', 'giving', 'gl', 'gm', 'gmt', 'gn', 'go', 'goes', 'going', 'gone', 'good', 'goods', 'got', 'gotten', 'gov', 'gp', 'gq', 'gr', 'great', 'greater', 'greatest', 'greetings', 'group', 'grouped', 'grouping', 'groups', 'gs', 'gt', 'gu', 'gw', 'gy', 'h', 'had', "hadn't", 'hadnt', 'half', 'happens', 'hardly', 'has', 'hasn', "hasn't", 'hasnt', 'have', 'haven', "haven't", 'havent', 'having', 'he', "he'd", "he'll", "he's", 'hed', 'hell', 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'herse\xe2\x80\x9d', 'hes', 'hi', 'hid', 'high', 'higher', 'highest', 'him', 'himself', 'himse\xe2\x80\x9d', 'his', 'hither', 'hk', 'hm', 'hn', 'home', 'homepage', 'hopefully', 'how', "how'd", "how'll", "how's", 'howbeit', 'however', 'hr', 'ht', 'htm', 'html', 'http', 'hu', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'i.e.', 'id', 'ie', 'if', 'ignored', 'ii', 'il', 'ill', 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'inc.', 'indeed', 'index', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'inside', 'insofar', 'instead', 'int', 'interest', 'interested', 'interesting', 'interests', 'into', 'invention', 'inward', 'io', 'iq', 'ir', 'is', 'isn', "isn't", 'isnt', 'it', "it'd", "it'll", "it's", 'itd', 'itll', 'its', 'itself', 'itse\xe2\x80\x9d', 'ive', 'j', 'je', 'jm', 'jo', 'join', 'jp', 'just', 'k', 'ke', 'keep', 'keeps', 'kept', 'keys', 'kg', 'kh', 'ki', 'kind', 'km', 'kn', 'knew', 'know', 'known', 'knows', 'kp', 'kr', 'kw', 'ky', 'kz', 'l', 'la', 'large', 'largely', 'last', 'lately', 'later', 'latest', 'latter', 'latterly', 'lb', 'lc', 'least', 'length', 'less', 'lest', 'let', "let's", 'lets', 'li', 'like', 'liked', 'likely', 'likewise', 'line', 'little', 'lk', 'll', 'long', 'longer', 'longest', 'look', 'looking', 'looks', 'low', 'lower', 'lr', 'ls', 'lt', 'ltd', 'lu', 'lv', 'ly', 'm', 'ma', 'made', 'mainly', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', "mayn't", 'maynt', 'mc', 'md', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'member', 'members', 'men', 'merely', 'mg', 'mh', 'microsoft', 'might', "might've", "mightn't", 'mightnt', 'mil', 'mill', 'million', 'mine', 'minus', 'miss', 'mk', 'ml', 'mm', 'mn', 'mo', 'more', 'moreover', 'most', 'mostly', 'move', 'mp', 'mq', 'mr', 'mrs', 'ms', 'msie', 'mt', 'mu', 'much', 'mug', 'must', "must've", "mustn't", 'mustnt', 'mv', 'mw', 'mx', 'my', 'myself', 'myse\xe2\x80\x9d', 'mz', 'n', 'na', 'name', 'namely', 'nay', 'nc', 'nd', 'ne', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needed', 'needing', "needn't", 'neednt', 'needs', 'neither', 'net', 'netscape', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nf', 'ng', 'ni', 'nine', 'ninety', 'nl', 'no', 'no-one', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'np', 'nr', 'nu', 'null', 'number', 'numbers', 'nz', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'older', 'oldest', 'om', 'omitted', 'on', 'once', 'one', "one's", 'ones', 'only', 'onto', 'open', 'opened', 'opening', 'opens', 'opposite', 'or', 'ord', 'order', 'ordered', 'ordering', 'orders', 'org', 'other', 'others', 'otherwise', 'ought', "oughtn't", 'oughtnt', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'pa', 'page', 'pages', 'part', 'parted', 'particular', 'particularly', 'parting', 'parts', 'past', 'pe', 'per', 'perhaps', 'pf', 'pg', 'ph', 'pk', 'pl', 'place', 'placed', 'places', 'please', 'plus', 'pm', 'pmid', 'pn', 'point', 'pointed', 'pointing', 'points', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'pr', 'predominantly', 'present', 'presented', 'presenting', 'presents', 'presumably', 'previously', 'primarily', 'probably', 'problem', 'problems', 'promptly', 'proud', 'provided', 'provides', 'pt', 'put', 'puts', 'pw', 'py', 'q', 'qa', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'reasonably', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'reserved', 'respectively', 'resulted', 'resulting', 'results', 'right', 'ring', 'ro', 'room', 'rooms', 'round', 'ru', 'run', 'rw', 's', 'sa', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sb', 'sc', 'sd', 'se', 'sec', 'second', 'secondly', 'seconds', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'sees', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'seventy', 'several', 'sg', 'sh', 'shall', "shan't", 'shant', 'she', "she'd", "she'll", "she's", 'shed', 'shell', 'shes', 'should', "should've", 'shouldn', "shouldn't", 'shouldnt', 'show', 'showed', 'showing', 'shown', 'showns', 'shows', 'si', 'side', 'sides', 'significant', 'significantly', 'similar', 'similarly', 'since', 'sincere', 'site', 'six', 'sixty', 'sj', 'sk', 'sl', 'slightly', 'sm', 'small', 'smaller', 'smallest', 'sn', 'so', 'some', 'somebody', 'someday', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'sr', 'st', 'state', 'states', 'still', 'stop', 'strongly', 'su', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure', 'sv', 'sy', 'system', 'sz', 't', "t's", 'take', 'taken', 'taking', 'tc', 'td', 'tell', 'ten', 'tends', 'test', 'text', 'tf', 'tg', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thatll', 'thats', 'thatve', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'therell', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'thereve', 'these', 'they', "they'd", "they'll", "they're", "they've", 'theyd', 'theyll', 'theyre', 'theyve', 'thick', 'thin', 'thing', 'things', 'think', 'thinks', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thought', 'thoughts', 'thousand', 'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'till', 'tip', 'tis', 'tj', 'tk', 'tm', 'tn', 'to', 'today', 'together', 'too', 'took', 'top', 'toward', 'towards', 'tp', 'tr', 'tried', 'tries', 'trillion', 'truly', 'try', 'trying', 'ts', 'tt', 'turn', 'turned', 'turning', 'turns', 'tv', 'tw', 'twas', 'twelve', 'twenty', 'twice', 'two', 'tz', 'u', 'ua', 'ug', 'uk', 'um', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'upwards', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'uucp', 'uy', 'uz', 'v', 'va', 'value', 'various', 'vc', 've', 'versus', 'very', 'vg', 'vi', 'via', 'viz', 'vn', 'vol', 'vols', 'vs', 'vu', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', 'wasn', "wasn't", 'wasnt', 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'web', 'webpage', 'website', 'wed', 'welcome', 'well', 'wells', 'went', 'were', 'weren', "weren't", 'werent', 'weve', 'wf', 'what', "what'd", "what'll", "what's", "what've", 'whatever', 'whatll', 'whats', 'whatve', 'when', "when'd", "when'll", "when's", 'whence', 'whenever', 'where', "where'd", "where'll", "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whim', 'whither', 'who', "who'd", "who'll", "who's", 'whod', 'whoever', 'whole', 'wholl', 'whom', 'whomever', 'whos', 'whose', 'why', "why'd", "why'll", "why's", 'widely', 'width', 'will', 'willing', 'wish', 'with', 'within', 'without', 'won', "won't", 'wonder', 'wont', 'words', 'work', 'worked', 'working', 'works', 'world', 'would', "would've", 'wouldn', "wouldn't", 'wouldnt', 'ws', 'www', 'x', 'y', 'ye', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'youll', 'young', 'younger', 'youngest', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve', 'yt', 'yu', 'z', 'za', 'zero', 'zm', 'zr']

stopWords = ["each","has", "had", "having", "do", "does", "did", "doing", "few", "more", "most", "other", "some", "such", "no","about", "against", "between", "into", "through", "during", "before","i", "me", "my", "myself", "we", "our", "ours","and", "but", "if",  "so", "than", "too", "very", "s", "t", "can", "will", "just", "or", "because", "as", "until", "while", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",  "of", "at", "by", "for", "with",  "after", "above","her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "have",  "a", "an", "the",  "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",  "nor",  "am", "is", "are", "was", "were", "be", "been", "being","not", "only", "own", "same","don", "should", "now"]

"""Initialization"""
# all correct classes
POSITIVE = 'positive'
NEGATIVE = 'negative'
DECEPTIVE = 'deceptive'
TRUTHFUL = 'truthful'
# classes = ["pos", "neg", "dec", "tru"]
classes = [POSITIVE, NEGATIVE, DECEPTIVE, TRUTHFUL]

POS_DEC = 'positive_deceptive'
NEG_DEC = 'negative_deceptive'
POS_TRU = 'postiive_truthful'
NEG_TRU = 'negative_truthful'
# classes = ["pos", "neg", "dec", "tru"]
mixedClasses = [POS_DEC, NEG_DEC, POS_TRU, NEG_TRU]

def main(input_path):
    """ Data path"""
    # base path for testing data
    # input_path = "./op_spam_test/"
    # input_path = os.path.dirname(os.path.realpath(__file__)) + "\op_spam_test_data\\"
    print("Root directory of the testing data: ", input_path)

    # #paths ot the testing data
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
    # path_directory = { pos_dec: [POSITIVE, DECEPTIVE],
    #                    neg_dec: [NEGATIVE, DECEPTIVE],
    #                    pos_tru: [POSITIVE, TRUTHFUL],
    #                    neg_tru: [NEGATIVE, TRUTHFUL]}
    #
    # path_directory = { POS_DEC: pos_dec,
    #                    NEG_DEC: neg_dec,
    #                    POS_TRU: pos_tru,
    #                    NEG_TRU: neg_tru}

    # print("Path Directory of each class: ", path_directory)


    # evaluation score
    score = {}

    """ Utility functions"""
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

    # To deal with situation where there is no token presence in the given class' cond prob
    def ifNoTokenInDict(condProbDict, token):
        # if the word is in the dict, return its conditional probability
        if token in condProbDict:
            return condProbDict[token]
        # if not return zero
        else:
            return 0

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

    # classify to obtain classes based on given file path
    def fourClassClassifier(fileName):
        # get the tokens from file after processing
        fileTokens = processFile(fileName)

        # add the tokens of the test file to dict
        fileDict = {}
        fileDict = addTokensToDict(fileDict, fileTokens)

        # loop through each class
        for category in classes:
            # get the score for each class by taking log10(priors)
            score[category] = math.log(priorProb[category], 10)

            # loop over each token USING LOG TO CALCULATE SCORE: multiply becomes sum
            for token in fileDict:
                if category == POSITIVE:
                    isTokenPresent = ifNoTokenInDict(condProbPos, token)
                    if isTokenPresent == 0: # that token does not belong to this class
                        continue  # does not change score
                    else:  # update the score if the token is present in the cond prob
                        score[category] += math.log(isTokenPresent, 10) #math.log(condProbPos[token])

                elif category == NEGATIVE:
                    isTokenPresent = ifNoTokenInDict(condProbNeg, token)
                    if isTokenPresent == 0:
                        continue
                    else:
                        score[category] += math.log((condProbNeg[token]), 10)

                elif category == DECEPTIVE:
                    isTokenPresent = ifNoTokenInDict(condProbDec, token)
                    if isTokenPresent == 0:
                        continue
                    else:
                        score[category] += math.log((condProbDec[token]), 10)

                elif category == TRUTHFUL:
                    isTokenPresent = ifNoTokenInDict(condProbTru, token)
                    if isTokenPresent == 0:
                        continue
                    else:
                        score[category] += math.log((condProbTru[token]), 10)
        # print("Score: ", score)
        return score


    """ Create output file"""
    # insert model accuracy to evaluate model
    scorePosNeg = 0
    totalScorePos = 0

    # match output file name with the given format
    output_file = "nboutput.txt"
    outputFile = open(output_file, "w")
    output = ""
    # Loop through each testing directory: key = correct classification, value = [correctClass1, correctClass2]
    # get all the files' directories
    filePaths = glob.glob(os.path.join(sys.argv[-1], '*/*/*/*.txt'))
    # print("Files: ", filePaths)
    for fileName in filePaths:
        # get the score
        score4Class = fourClassClassifier(fileName)
        # print(score4Class)

        # initialize output
        output = []

        # compare score between dec-tru and pos-neg
        if score4Class[DECEPTIVE] > score4Class[TRUTHFUL]:
            class1 = "deceptive"
        else:
            class1 = "truthful"

        if score4Class[POSITIVE] > score4Class[NEGATIVE]:
            class2 = "positive"
        else:
            class2 = "negative"

        # Check accuracy

        # reconstruct output as given format
        output = class1 + " " + class2 + " " + fileName + "\n"
        # print(output)

        # write to output file
        outputFile.write(output)



    # for key, value in path_directory.items():
    #     print("Testing class: ", key)
    #
    #     # loop through each directory path belong to each class
    #     for path in value:
    #         print("Testing class ", key, "with files in path ", path)
    #         # os.path.abspath(os.path.join(dir, fileName)), sep = '\n')
    #         for root, dirnames, fileNames in os.walk(path):
    #             # filter for txt files
    #             for fileName in fnmatch.filter(fileNames, '*.txt'):
    #                 # reconstruct file path
    #                 # fileName = os.path.abspath(os.path.join(root, fileName))
    #                 fileName = os.path.join(root, fileName)
    #                 print("currently reading: ", fileName)
    #
    #                 # get the score
    #                 score4Class = fourClassClassifier(fileName)
    #                 # print(score4Class)
    #
    #                 # initialize output
    #                 output = []
    #
    #                 # compare score between dec-tru and pos-neg
    #                 if score4Class[DECEPTIVE] > score4Class[TRUTHFUL]:
    #                     class1 = "deceptive"
    #                 else:
    #                     class1 = "truthful"
    #
    #                 if score4Class[POSITIVE] > score4Class[NEGATIVE]:
    #                     class2 = "positive"
    #                 else:
    #                     class2 = "negative"
    #
    #                 # Check accuracy
    #
    #                 # reconstruct output as given format
    #                 output = class1 + " " + class2 + " " + fileName + "\n"
    #                 # print(output)
    #
    #                 # write to output file
    #                 outputFile.write(output)

    # close output file
    outputFile.close()


""" RUN THE PROGRAM"""
# inPath = os.path.dirname(os.path.realpath(__file__)) + "\\op_spam_testing_data\\"

# for command line invoke: python ./nbclassify.py ./op_spam_testing_data
inPath = str(sys.argv[-1])
print("inPath: ", inPath)

main(inPath)