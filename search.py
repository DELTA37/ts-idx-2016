import argparse
#import pandas as pd
import bisect
import struct
from collections import defaultdict, namedtuple, OrderedDict
from docreader import varbyte_decode, simple9_decode
import pickle
import cPickle as pickle
import numpy as np

seek_t = namedtuple('seek_t', ['diff', 'size'])

class HashTable:
    def __init__(self, filename):
        self.lst = []
        self.values = []
        with open(filename, 'r') as fp:
            '''
            for line in fp:
                line = line.strip().split(':')
                line[0] = int(line[0])
                line[1] = int(line[1])
                try:
                    line[2] = int(line[2])
                except:
                    line[2] = -1
                self.lst.append(line[0])
                self.values.append(seek_t(line[1], line[2]))
            '''
            for s in iter(lambda : fp.read(16), ''):
                hsh, diff, size = struct.unpack('q', s[0:8])[0], struct.unpack('i', s[8:12])[0], struct.unpack('i', s[12:])[0]
                self.lst.append(hsh)
                self.values.append(seek_t(diff, size))

    def __getitem__(self, h):
        i = bisect.bisect(self.lst, h) - 1
        if self.lst[i] != h:
            raise RuntimeError
        return self.values[i]
        
#df = None
hash_table = None
f = None
docids = {}
docids_set = set()

def Init():
    global df, f, docids, docids_set, hash_table
    #df = pd.read_csv('./dict.txt', sep=':', names=['hash', 'diff', 'size']).set_index('hash')
    hash_table = HashTable('./dict.txt')
    f = open('./index.txt', 'r')
    docids = pickle.load(open('./docids.txt', 'r'))
    docids_set = set(docids.keys())
    '''
    for docid in docids:
        print(docid)
        print(docids[docid])
    #'''

def getDocidsByKey(key, decoding):
    global hash_table
    #diff = df.loc[hash(key)]['diff']
    #size = df.loc[hash(key)]['size']
    try:
        s = hash_table[hash(key)]
    except:
        return set()
    diff = s.diff
    size = s.size
    f.seek(diff)

    try:
        size = int(size)
    except:
        size = -1

    if size > 0:
        s = bytearray(f.read(int(size)))
    else:
        s = bytearray(f.read())
    
    s = decoding(s)
    lst = parse(s, key)

    return lst

def parse(lst, key):
    pt = list(map(ord, key))
    res = []
    prev = -1
    n = len(lst)
    i = 0
    while(i < n):
        if lst[i] == 10:
            key = lst[prev+1:i]
            if key == pt:
                l = lst[i+1]
                docids = lst[i+2:i+2+l]
                return set(docids)
        i += 1
    return set()


class Op(object):
    def __init__(self):
        self.who = ''


class Term(Op):
    def __init__(self, s, neg=0):
        super(Term, self).__init__()
        self.val = s
        self.neg = neg
        self.who = 'term'
    def __str__(self):
        return u'!'*self.neg + self.val


class And(Op):
    def __init__(self, lst, neg=0):
        super(And, self).__init__()
        self.lst = lst
        self.neg = neg
        self.who = 'and'
    def __len__(self):
        return len(self.lst)
    def __str__(self):
        return u'!'*self.neg + u'&'.join(list(map(lambda x : x.__str__(), self.lst)))

class Or(Op):
    def __init__(self, lst, neg=0):
        super(Or, self).__init__();
        self.lst = lst
        self.neg = neg
        self.who = 'or'
    def __len__(self):
        return len(self.lst)

    def __str__(self):
        return u'!'*self.neg + u'|'.join(list(map(lambda x : x.__str__(), self.lst)))

def getMainOp(s):
    s = s.strip()
    s.replace(' ', '')
    s.replace('\n', '')
    s.replace('\t', '')
    if len(s) == 0:
        return None
    if s.find('(') == -1:
        if s.find('|') == -1:
            if s.find('&') == -1:
                if s[0] == '!':
                    return Term(s[1:], neg=1)
                else:
                    return Term(s)
            lst = s.split('&')
            lst = list(map(getMainOp, lst))
            return And(lst)
        else:
            lst = s.split('|')
            lst = list(map(getMainOp, lst))
            return Or(lst)
    else:
        cur_br = 0
        prev_or = -1
        lst = []
        for i in range(len(s)):
            if s[i] == '(':
                cur_br += 1
            elif s[i] == ')':
                cur_br -= 1
            elif s[i] == '|':
                if cur_br == 0:
                    lst.append(s[prev_or+1:i])
                    prev_or = i
        lst.append(s[prev_or+1:])
        if prev_or == -1:
            cur_br = 0
            prev_and = -1
            lst = []
            for i in range(len(s)):
                if s[i] == '(':
                    cur_br += 1
                elif s[i] == ')':
                    cur_br -= 1
                elif s[i] == '&':
                    if cur_br == 0:
                        lst.append(s[prev_and+1:i])
                        prev_and = i
            lst.append(s[prev_and+1:])
            if prev_and == -1:
                if s[0] == '(' and s[-1] == ')':
                    return getMainOp(s[1:-1])
                elif s[0] == '!':
                    res = getMainOp(s[2:-1])
                    res.neg = 1 - res.neg
                    return res
            else:
                lst = list(map(getMainOp, lst))
                return And(lst)
        else:
            lst = list(map(getMainOp, lst))
            return Or(lst)


def queryTransform(op):
    if op == None:
        return None
    if op.who == 'term':
        return op
    elif op.who == 'or':
        b = 1
        for i in range(len(op.lst)):
            op.lst[i] = queryTransform(op.lst[i])
            b *= op.lst[i].neg
        if b:
            for i in range(len(op.lst)):
                op.lst[i].neg = 0
            op = And(op.lst, 1 - op.neg)
        return op
    elif op.who == 'and':
        b = 1
        for i in range(len(op.lst)):
            op.lst[i] = queryTransform(op.lst[i])
            b *= op.lst[i].neg
        if b:
            for i in range(len(op.lst)):
                op.lst[i].neg = 0
            op = Or(op.lst, 1 - op.neg)

        op.lst = sorted(op.lst, key=lambda x : x.neg)
        return op

def queryExecute(op, encoding):
    if op == None:
        return set()
    if op.who == 'term':
        lst = getDocidsByKey(op.val, encoding)
        if op.neg == 1:
            #lst = [x for x in docids_set if x not in lst]
            lst = docids_set.difference(lst)
        return lst
    elif op.who == 'or':
        res = set()
        for i in range(len(op.lst)):
            #res.extend(queryExecute(op.lst[i], encoding))
            res = res.union(queryExecute(op.lst[i], encoding))
        return res
    elif op.who == 'and':
        res = queryExecute(op.lst[0], encoding)
        for i in range(1, len(op.lst)):
            q = queryExecute(op.lst[i], encoding)
            #res = [x for x in res if x in q]
            res = res.intersection(q)
        return res
    else:
        raise RuntimeError



class Query:
    def __init__(self, s, encoding):
        self.s = s
        self.encoding = encoding
        self.ops = getMainOp(s)
    def execute(self):
        self.ops = queryTransform(self.ops)
        res = list(queryExecute(self.ops, self.encoding))
        res = sorted(res)
        return list(map(lambda x : docids[x], res))
        #return res


        
if __name__ == '__main__':
    Init()
    while (True):
        try:
            s = raw_input()
            print(s)
            s = s.decode('utf8').lower()
        except (EOFError):
            break
        a = pickle.load(open('coding_type.txt', 'r'))
        if a == 'varbyte':
            encoding = varbyte_decode
        elif a == 'simple9':
            encoding = simple9_decode
        else:
            raise NotImplemented
        q = Query(s, encoding)
        res = q.execute()
        print(len(res))
        for i in range(len(res)):
            print(res[i])




