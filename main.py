import fire

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
    deps = []
    tmp_deps = []
    # get all deps
    for raw_line in raw_lines:
        if 'EOS' in raw_line:
            deps.append(tmp_deps)
            tmp_deps = []
        elif raw_line.startswith('*'):
            tmp_deps.append(raw_line)

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



def fourty():
    lines, deps = fourty_beta()
    sep_lines = []
    tmp_lines = []
    for line in lines:
        if line.surface != '<EOS>' and line.surface != '<SEP>':
            tmp_lines.append(line)
        else:
            sep_lines.append(tmp_lines)
            tmp_lines = []
    
    return sep_lines, deps


class Chunk(object):
    
    def __init__(self):
        pass


def fourtyone():
    sep_lines, deps = fourty()
    print(sep_lines[0])
    print(deps[len(deps)-1])



fire.Fire()
