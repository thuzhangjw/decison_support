import jieba
import pandas as pd
import re
import gensim
import os


def text_keyword_mapping(series):
    table = {
            '稳定性心绞痛': '稳定型心绞痛',
            '稳定心绞痛': '稳定型心绞痛',
            '抬高性心肌梗死': '抬高型心肌梗死',
            '冠状动脉动脉粥样硬化性心脏病': '冠状动脉粥样硬化性心脏病',
            '冠状动脉粥样硬化心脏病': '冠状动脉粥样硬化性心脏病'
            }

    def repl(s):
        res = s
        for k in table:
            res = res.replace(k, table[k])
        return res

    for i in range(len(series)):
        series[i] = repl(series[i])


def text_participle(series):
    jieba.load_userdict('./ds/my_dict_from_other_source.txt')
    jieba.load_userdict('./ds/mydict.txt')
    rs = [re.compile('[(（][^(（]*[)）]'), re.compile('患者自发病以来.*。')]
    stopwords = set()
    neg_words=['无']
    
    with open('./ds/stopword-full.dic', 'r') as f:
        lines = f.readlines()
        for word in lines:
            stopwords.add(word.strip())


    def load_suggest_freq():
        if os.path.exists('./ds/suggest_freq.txt'):
            f = open('./ds/suggest_freq.txt', 'r')
            lines = f.readlines()
            for line in lines:
                words = line.split(' ')
                jieba.suggest_freq((words[0], words[1]), True)
            f.close()

    def add_negtive_word(wordslist):
        preprocessd = list(map(lambda x: [x, '、'][x == '及'], wordslist))
        pos = []
        for idx, word in enumerate(preprocessd):
            for nw in neg_words:
                if word == nw:
                    for i in range(idx+1, len(preprocessd)-1):
                        if preprocessd[i] == '、' and preprocessd[i+1] not in neg_words:
                            pos.append((i+1, nw))
                        elif preprocessd[i] == ',' or preprocessd[i] == '，' or preprocessd[i] == '.' or preprocessd[i] == '。':
                            break
        res = preprocessd
        for idx, p in enumerate(pos):
            res.insert(idx + p[0], p[1])
        return res 


    load_suggest_freq()
    sentence_list = []
    word_list_list = []
    for text in series:
        for r in rs:
            text = r.sub('', text)
        sentences = text.split('。')
        newdoc = ''
        for s in sentences:
            news = ''
            if s == '':
                continue
            seg_list = jieba.lcut(s)
            processd_list = add_negtive_word(seg_list)
            for word in processd_list:
                if word.strip() not in stopwords:
                    news += word.strip() + ' '
            news += '。 '
            word_list_list.append(news.strip('。 '))
            newdoc += news 
        newdoc = newdoc.strip()
        sentence_list.append(newdoc)
    
    disease_his_df = pd.DataFrame({'disease_his': sentence_list})
    print(' save words corups')
    with open('./ds/corups.txt', 'w') as f:
        for s in word_list_list:
            f.write(s + '\n')
    
    return disease_his_df  


def word2vec(path_to_save):
    print(' pre train word embedding vector')
    sentences = gensim.models.word2vec.LineSentence('./ds/corups.txt')
    model = gensim.models.word2vec.Word2Vec(sentences, min_count=0, workers=8, iter=10, size=100, window=5)
    model.save(path_to_save)
    return model 

