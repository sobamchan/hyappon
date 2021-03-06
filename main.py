import fire
from sobamchan import sobamchan_utility
from sobamchan import sobamchan_vocabulary

util = sobamchan_utility.Utility()

def one(s='stressed'):
    print(s[::-1])

def two():
    pa = 'パトカー'
    ta = 'タクシー'
    for p, t in zip(pa, ta):
        print(p, t)

def three():
    s = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    li = s.split()
    for i in li:
        print(len(i))

def four():
    s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
    li = [1,5,6,7,8,9,15,16,19]
    li = [i-1 for i in li]
    for i, j in enumerate(s.split()):
        if i in li:
            print(j[0])
        else:
            print(j[:2])

def ngram(s, n):
    result = []
    i = 0
    for _ in range(len(s)):
        result.append(s[i:i+n])
        i += 1
    return result

def five():
    s = 'I am a NLPer'
    result = ngram(s.split(), 2)
    print(result)

    s = [i for i in s]
    result = ngram(s, 2)
    print(result)

def six():
    a = 'paraparaparadise'
    b = 'paragraph'
    ab = set(ngram(a, 2))
    bb = set(ngram(b, 2))
    print(ab&bb)
    print(ab|bb)

def seven(x, y, z):
    print('{}時の{}は{}.'.format(x, y, z))

def eight():
    pass

def nine():
    import random
    s = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
    li = s.split()
    mid = []
    for word in li[1:-1]:
        if len(word) < 4:
            mid.append(word)
        else:
            mid.append(''.join(list(random.sample(word, len(word)))))

    print(li[0] + ' '.join(mid) + li[len(li)-1])

def ten():
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    print(len(text))

def eleven():
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    text = [s.replace('\t', ' ') for s in text]
    print(text)

def twenteen():
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    col1 = [line.split('\t')[0] for line in text]
    col2 = [line.split('\t')[1] for line in text]
    print(col1)
    print(col2)

def thirteen():
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    col1 = [line.split('\t')[0] for line in text]
    col2 = [line.split('\t')[1] for line in text]
    result = ['{}\t{}'.format(one, two) for one, two in zip(col1, col2)]
    print(result)

def fourteen(n):
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    print(text[:n])

def fifteen(n):
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    print(text[-n:])

def sixteen(n):
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    i = 0
    result = []
    for _ in text:
        result.append(text[i:i+n])
        i += n
    result = list(filter(lambda x:len(x)>1, result))
    print(result)

def seventeen():
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    col1 = [line.split('\t')[0] for line in text]
    uniq_col1 = list(set(col1))
    print(uniq_col1)

def eightteen():
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    col3 = [line.split('\t')[2] for line in text]
    sorted_col3 = list(reversed(sorted(col3)))
    print(sorted_col3)

def nineteen():
    import collections
    with open('./hightemp.txt', 'r') as f:
        text = f.readlines()
    col1 = [line.split('\t')[0] for line in text]
    counter = collections.Counter(col1)
    print(counter)

def get_wiki():
    import json
    with open('./jc.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d

def get_england():
    d = get_wiki()
    england = list(filter(lambda x:'イギリス' in x['text'], d))
    return england

def twenty():
    import json
    with open('./jc.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    england = list(filter(lambda x:'イギリス' in x['text'], d))
    print(len(england))

def twentyone():
    e = get_england()
    texts = [content['text'] for content in e]
    category_texts = list(filter(lambda text: 'category' in text, texts))
    category_lines = []
    for category_text in category_texts:
        for line in category_text.split('\n'):
            if 'Category' in line:
                category_lines.append(line)
                # print(line)
                # print('------')

    return category_lines

def twentytwo():
    import re
    category_lines = twentyone()
    ptn = r'^.+Category:(.+)\]\]$'
    r = re.compile(ptn)
    for category_line in category_lines:
        m = r.search(category_line)
        if m:
            print(m.group(1))

def twentythree():
    # TODO
    e = get_england()
    texts = [content['text'] for content in e]
    for text in texts:
        for line in text.split('\n'):
            if 'Section' in line:
                print(line)
                print(('-----'))

def twentyfour():
    pass



def get_neko():
    with open('./neko.txt', 'r') as f:
        d = f.readlines()
    return d

def thirty():
    result = []
    with open('./neko.txt.mecab', 'r') as f:
        lines = f.readlines()
    for line in lines:
        if 'EOS' in line:
            continue
        surface, functions = line.split('\t')
        functions = functions.split(',')
        result.append({
            'surface': surface,
            'base': functions[-3],
            'pos': functions[0],
            'pos1': functions[1]
            })
    return result

def thirtyone():
    d = thirty()
    verb = list(filter(lambda x:x['pos'] == '動詞', d))
    return verb

def thirtytwo():
    verbs = thirtyone()
    origins = [verb['base'] for verb in verbs]
    return origins

def thirtythree():
    d = thirty()
    r = list(filter(lambda x:x['pos1'].startswith('サ変') and x['pos'] == '名詞', d))
    return r

def thirtyfour():
    words = thirty()
    tri = words[:3]
    no_nouns = []
    for word in words[3:]:
        tri = tri[1:] + [word]
        if tri[1]['surface'] == 'の' and tri[0]['pos'] == tri[0]['pos'] == '名詞':
            no_nouns.append(tri)
    return no_nouns

def thirtyfive():
    words = thirty()
    comb = words[:2]
    results = []
    for word in words[2:]:
        comb = comb[1:] + [word]
        if comb[0] == comb[1]:
            results.append(comb)
    return results

def thirtysix(top_n=None):
    import collections
    words = thirty()
    counter = collections.Counter([word['surface'] for word in words])
    if top_n is None:
        top_n = len(counter)
    return counter.most_common(top_n)

def thirtyseven():
    # TODO 文字化けする
    import matplotlib.pyplot as plt
    words = thirtysix(10)
    X_label = [word[0] for word in words]
    Y = [word[1] for word in words]
    X = range(len(Y))
    plt.xticks(X, X_label)
    plt.plot(X, Y)
    plt.show()

def thirtyeight():
    import matplotlib.pyplot as plt
    words = thirtysix(100)
    X_label = [word[0] for word in words]
    Y = [word[1] for word in words]
    X = range(len(Y))
    plt.hist(Y)
    plt.show()

def thirtynine():
    import matplotlib.pyplot as plt
    import math
    words = thirtysix(100)
    X = list(range(len(words)))
    X = [math.log(x+1) for x in X]
    Y = [math.log(word[1]) for word in words]
    plt.plot(X, Y)
    plt.show()


class Morph(object):

    def __init__(self, surface, base=None, pos=None, pos1=None):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1
    
    def __str__(self):
        return self.surface

    def __repr__(self):
        return self.__str__()


def fourty_beta():
    with open('./neko.txt.cabocha', 'r') as f:
        raw_lines = f.readlines()
    lines = []
    deps = []
    for raw_line in raw_lines:
        if raw_line.startswith('*') :
            deps.append(raw_line)
            lines.append(Morph('<SEP>'))
            continue
        if 'EOS' in raw_line:
            deps.append('<EOS>')
            lines.append(Morph('<EOS>'))
            continue
        surface, features = raw_line.split('\t')
        features = features.split(',')
        lines.append(Morph(
            surface, features[-2], features[0], features[1]
            ))

    return lines, deps

def get_morph(line):
    surface, features = line.split('\t')
    features = features.split(',')

    return (Morph(surface, features[-2], features[0], features[1]))

def cabocha_parser():
    with open('./neko.txt.cabocha', 'r') as f:
        raw_lines = f.readlines()
    deps_ids = []
    tmp_deps_ids = []
    deps = []
    tmp_deps = []
    # get all deps
    for raw_line in raw_lines:
        if 'EOS' in raw_line:
            deps.append(tmp_deps)
            deps_ids.append(tmp_deps_ids)
            tmp_deps = []
            tmp_deps_ids = []
        elif raw_line.startswith('*'):
            tmp_deps.append(raw_line.split()[2].replace('D', ''))
            tmp_deps_ids.append(raw_line.split()[1])

    lines = []
    tmp_line = []
    morphs = []
    # get all morphs
    for i, raw_line in enumerate(raw_lines):
        if 'EOS' in raw_line:
            lines.append(tmp_line)
            if len(morphs) != 0:
                tmp_line.append(morphs)
                morphs = []
            tmp_line = []
            morphs = []
            continue
        if raw_line.startswith('*'):
            if len(morphs) != 0:
                tmp_line.append(morphs)
                morphs = []
        else:
            morphs.append(get_morph(raw_line))

    return lines, deps, deps_ids


def fourty():
    lines, deps, _ = cabocha_parser()
    return lines, deps


class Chunk(object):
    
    def __init__(self, morphs, dst, dep_id, srcs):
        self.morphs = morphs
        self.dst = dst
        self.dep_id = dep_id
        self.srcs = srcs

    def __str__(self):
        rlt = ''
        for morph in self.morphs:
            rlt += ' ' + morph.surface
        rlt += 'dst: {}, srcs: {}'.format(self.dst, self.srcs)
        return rlt

    def __repr__(self):
        return self.__str__()

    def get_surfaces(self):
        surfaces = ''
        for morph in self.morphs:
            surfaces += morph.surface
        return surfaces

    def has_noun(self):
        noun_morphs = list(filter(lambda morph:morph.pos == '名詞', self.morphs))
        return len(noun_morphs) > 0

    def has_verb(self):
        verb_morphs = list(filter(lambda morph:morph.pos == '動詞', self.morphs))
        return len(verb_morphs) > 0

def fourtyone():
    lines, deps, deps_ids = cabocha_parser()
    chunks = []
    for line, dep, deps_id in zip(lines, deps, deps_ids):
        dep = list(map(int, dep))
        dep_stock = []
        tmp_chunks = []
        for i in range(len(line)):
            morphs = line[i]
            dst = dep[i]
            dep_id = deps_id[i]
            srcs = list(filter(lambda x:x == dst, dep[:i]))
            tmp_chunks.append(Chunk(morphs, dst, dep_id, srcs))
        chunks.append(tmp_chunks)

    return chunks

def fourtytwo():
    chunk_list = fourtyone()
    relations = []
    for chunks in chunk_list:
        for i, chunk in enumerate(chunks):
            dst_chunks = list(filter(lambda x: chunk.dst in x.srcs, chunks[i+1:]))
            if len(dst_chunks) != 0:
                dst_surfaces = ''
                for dst_chunk in dst_chunks:
                    dst_surfaces += dst_chunk.get_surfaces()
                relations.append('{} \t {}'.format(chunk.get_surfaces(), dst_surfaces))
    print(relations)

def fourtythree():
    chunk_list = fourtyone()
    relations = []
    for chunks in chunk_list:
        for i, chunk in enumerate(chunks):
            if chunk.has_noun():
                dst_chunks = list(filter(lambda x: chunk.dst in x.srcs, chunks[i+1:]))
                for dst_chunk in dst_chunks:
                    if dst_chunk.has_verb():
                        relations.append('{}\t{}'.format(chunk.get_surfaces(), dst_chunk.get_surfaces()))
                        break
    print(relations)

def fourtyfour():
    import pydot
    chunk_list = fourtyone()
    chunks = chunk_list[10]
    edges = []
    for chunk in chunks:
        edges.append((int(chunk.dep_id), chunk.dst))
    g=pydot.graph_from_edges(edges)
    g.write_jpeg('graph_from_edges_dot.jpg', prog='dot')

# TODO 45 - 49

def fifty():
    import re
    with open('./nlp.txt', 'r') as f:
        lines = f.readlines()
    ptn = r'(?<=[^A-Z].[.?|.!|/;]) +(?=[A-Z])'
    c = re.compile(ptn)
    result = []
    for line in lines:
        result += c.split(line)

    return result

def fiftyone():
    lines = fifty()
    for line in lines:
        for word in line.split():
            print(word)
        print('\n')

def fiftytwo():
    from stemming.porter2 import stem
    lines = fifty()
    for line in lines:
        for word in line.split():
            print(word, stem(word))
        print('\n')

def fiftythree():
    import xml.etree.ElementTree as ET
    tree = ET.parse('./nlp.txt.xml')
    root = tree.getroot()
    for sentences in root.iter('sentences'):
        for sentence in sentences.iter('tokens'):
            for token in sentence.iter('token'):
                print(token.find('word').text)

def fiftyfour():
    import xml.etree.ElementTree as ET
    tree = ET.parse('./nlp.txt.xml')
    root = tree.getroot()
    for sentences in root.iter('sentences'):
        for sentence in sentences.iter('tokens'):
            for token in sentence.iter('token'):
                word = token.find('word').text
                lemma = token.find('lemma').text
                pos = token.find('POS').text
                print('{}\t{}\t{}'.format(word, lemma, pos))

def fiftyfive():
    import xml.etree.ElementTree as ET
    tree = ET.parse('./nlp.txt.xml')
    root = tree.getroot()
    for sentences in root.iter('sentences'):
        for sentence in sentences.iter('tokens'):
            for token in sentence.iter('token'):
                word = token.find('word').text
                lemma = token.find('lemma').text
                pos = token.find('POS').text
                ner = token.find('NER').text
                if ner == 'PERSON':
                    print(word)

def fiftysix():
    import xml.etree.ElementTree as ET
    tree = ET.parse('./nlp.txt.xml')
    root = tree.getroot()
    # {sentence_id, start, end, text, represent_of}
    references = {}
    i = 0
    for coreference in root.iter('coreference'):
        for corefe in coreference.iter('coreference'):
            mentions = {}
            for mention in corefe.iter('mention'):
                if 'representative' in mention.attrib.keys():
                    mentions['rep'] = mention
                else:
                    if not 'hireps' in mentions.keys():
                        mentions['hireps'] = []
                    mentions['hireps'].append(mention)
            
            if len(mentions.keys()) != 0:
                for hirep in mentions['hireps']:
                    i += 1
                    k = int(hirep.find('sentence').text)
                    references[k] = {}
                    references[k]['replace_with'] = mentions['rep'].find('text').text
                    references[k]['start'] = hirep.find('start').text
                    references[k]['end'] = hirep.find('end').text
                    references[k]['directive'] = hirep.find('text').text

    for sentences in root.iter('sentences'):
        for sentence in sentences.iter('sentence'):
            sid = int(sentence.attrib['id'])
            if sid in references.keys():
                for tokens in sentences.iter('tokens'):
                    s = ' '.join([token.find('word').text for token in tokens.iter('token')])
                    reference = references[sid]
                    s.replace(reference['directive'], '[{}]({})'.format(reference['directive'], reference['replace_with']))
                    print(s)
            else:
                for tokens in sentences.iter('tokens'):
                    for token in tokens.iter('token'):
                        pass
                        # print(token.find('word').text, end=' ')
            print()

def fiftyseven():
    import pydot
    import xml.etree.ElementTree as ET
    tree = ET.parse('./nlp.txt.xml')
    root = tree.getroot()

    ldependencies = []

    for dependencies in root.iter('dependencies'):
        ldeps = []
        for dep in dependencies.iter('dep'):
            ldeps.append([int(dep.find('governor').attrib['idx']), int(dep.find('dependent').attrib['idx'])])
        ldependencies.append(ldeps)
    for i, ldependency in enumerate(ldependencies):
        g=pydot.graph_from_edges(ldependency)
        g.write_jpeg('{}.jpg'.format(i), prog='dot')

def fiftyeight():
    pass

def fiftynine():
    pass

def seventyone(word):
    stopwords = ['i', 'you']
    return not word.lower() in stopwords

def test(line='i like you'):
    print(list(filter(seventyone, line.split())))

def seventytwo():
    from stemming.porter2 import stem
    d = util.load_json('./posneg.json')
    parsed_d = []
    for line in d:
        tag, line = line.split('\t')
        line = ' '.join([stem(w).lower() for w in list(filter(seventyone, line.split()))])
        parsed_d.append('{}\t{}'.format(tag, line))

    return parsed_d

def get_onehot(wid, vocab_n):
    li = [0] * vocab_n
    li[wid] = 1
    return li

def sigmoid(z):
    import numpy as np
    return 1.0 / (1 + np.exp(-z))

def get_dataset(d_n=None):
    import numpy as np
    from tqdm import tqdm
    d = seventytwo()
    vocab = sobamchan_vocabulary.Vocabulary()
    [vocab.new(line.split('\t')[1]) for line in d]
    X = []
    Y = []
    print('building dataset')
    if d_n:
        d = d[:d_n]
    for line in tqdm(d):
        tag, line = line.split('\t')
        if '+' in tag:
            tag = 1
        else:
            tag = 0
        line = np.array([get_onehot(vocab.w2i[word], len(vocab)) for word in line.split()])
        vec = np.sum(line, axis=0)
        X.append(vec)
        Y.append(tag)
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, vocab

def seventythree():
    import numpy as np
    from tqdm import tqdm
    X, Y, vocab = get_dataset()
    N = len(X)
    W = np.random.rand(len(vocab))

    print('learning')
    eta = 0.1
    for _ in tqdm(range(50)):
        fail = 0
        for i in range(N):
            xn = X[i, :]
            yn = Y[i]
            predict = sigmoid(np.inner(W, xn))
            W -= eta * (predict - yn) * xn
        eta *= 0.9

    return X, Y, W, vocab

def seventyfour():
    # included in seventythree
    pass

def seventyfive():
    import numpy as np
    X, _, W, vocab = seventythree()
    sorted_args_W = np.argsort(W)
    for i in sorted_args_W[:10]:
        print(vocab.i2w[i])
    for i in sorted_args_W[-10:]:
        print(vocab.i2w[i])

def seventysix():
    import numpy as np
    X, Y, W, vocab = seventythree()
    correct = 0
    print('testing model')
    result = []
    for x, y in zip(X, Y):
        predict = sigmoid(np.inner(W, x))
        predict_bin = 1 if predict > 0.5 else 0
        correct = correct + 1 if y == predict_bin else correct
        result.append(('{}\t{}\t{}'.format(y, predict_bin, predict)))
    print('Accuracy: ', correct / len(X))

    return result

def seventyseven():
    results = seventysix()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for result in results:
        t, y, predict = result.split('\t')
        if t == 1 and y == 1:
            tp += 1
        if t == 1 and y == 0:
            tn += 1
        if t == 0 and y == 1:
            tn += 1
        if t == 0 and y == 0:
            fn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('F-measure: {}'.format((2*recall)/(recall+precision)))

def seventyeight():
    import numpy as np
    from tqdm import tqdm
    X, Y, vocab = get_dataset()
    train_N = int(len(X) * 0.5)
    train_X = X[:train_N]
    test_X = X[train_N:]
    train_Y = Y[:train_N]
    test_Y = Y[train_N:]
    train_N = len(train_X)
    test_N = len(test_X)

    W = np.random.rand(len(vocab))

    print('learning')
    eta = 0.1
    for _ in tqdm(range(50)):
        for i in range(train_N):
            xn = train_X[i, :]
            yn = train_Y[i]
            predict = sigmoid(np.inner(W, xn))
            W -= eta * (predict - yn) * xn
        eta *= 0.9

    correct = 0
    for i in range(test_N):
        xn = test_X[i, :]
        yn = test_Y[i]
        predict = sigmoid(np.inner(W, xn))
        predict_bin = 1 if predict > 0.5 else 0
        correct = correct + 1 if yn == predict_bin else correct
    print('test accuracy: {}'.format(correct/test_N))

def seventynine():
    pass

def eighty():
    from tqdm import tqdm
    import  re
    lines = util.readlines_from_filepath('./enwiki-20150112-400-r10-105752.txt')
    ptn = r'^[\.,!\?;:\(\)\[\]\'"]?(\w+)[\.,!\?;:\(\)\[\]\'"]?$'
    ptn_compiled = re.compile(ptn)
    tokens = [ptn_compiled.search(word.strip().lower()).group() if ptn_compiled.search(word.lower()) else '' for line in tqdm(lines) for word in line.strip().split()]
    tokens = list(filter(lambda x:len(x)>0, tokens))
    with open('parsed_enwiki.txt', 'w') as f:
        f.write(' '.join(tokens))

def eightyone():
    from tqdm import tqdm
    import re
    d = util.readlines_from_filepath('./parsed_enwiki.txt')[0]
    countries = util.readlines_from_filepath('./country_list.txt')
    countries = [country.strip().lower() for country in countries]
    countries = [country.split(';')[0] for country in countries]
    countries_space = [country.replace(' ', '_') for country in countries]
    for country, country_space in tqdm(zip(countries, countries_space)):
        r = re.compile(r'{}'.format(country))
        d = r.sub(country_space, d)
    with open('parsed_enwiki_spaced.txt', 'w') as f:
        f.write(d)

def eightytwo():
    import random
    from tqdm import tqdm
    d = util.readlines_from_filepath('./parsed_enwiki_spaced.txt')[0]
    N = len(d)
    li = []
    dsplit = d.split()
    for i, word in tqdm(enumerate(dsplit), total=len(dsplit)):
        window_n = random.choice([1,2,3,4,5])
        start = i - window_n if i - window_n > 0 else 0
        end = i + window_n if i - window_n <= N else N
        context = '\t'.join(dsplit[start:i] + dsplit[i:end])
        li.append('{}\t{}'.format(word, context))
    # util.save_json(li, './enwiki_context.json')
    return li
    
def eightythree():
    from tqdm import tqdm
    from collections import Counter
    # ds = util.load_json('./enwiki_context.json')
    ds = eightytwo()
    ftc = Counter()
    ft = Counter()
    fc = Counter()
    for d in tqdm(ds):
        word = d.split('\t')[0]
        context = '\t'.join(d.split('\t')[1:])
        ftc[d] += 1
        ft[word] += 1
        fc[context] += 1
    N = len(ftc)

    return ftc, ft, fc, N

def eightyfour():
    import math
    from tqdm import tqdm
    import pickle
    X = []
    ftc, ft, fc, N = eightythree()
    # ds = util.load_json('./enwiki_context.json')
    ds = eightytwo()

    words = list(ft.values())
    contexts = list(fc.values())

    for word in tqdm(words):
        x_word = []
        for context in contexts:
            d = '{}\t{}'.format(word, context)
            if ftc[d] > 10:
                if (N * ftc[d]) / (ft[word] * fc[context]) != 0:
                    ppmi = max(math.log((N * ftc[d]) / (ft[word] * fc[context])), 0) 
                else:
                    ppmi = 0
            else:
                ppmi = 0
            x_word.append(ppmi)
        X.append(x_word)

    print(len(X))
    print(len(X[0]))
    print(len(X[1]))
    pickle.dump(file='./X.pkl', obj=X)

fire.Fire()
