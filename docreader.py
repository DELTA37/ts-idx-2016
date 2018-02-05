#!/usr/bin/env python
import argparse
import document_pb2
import struct
import gzip
import sys
from doc2words import extract_words_doc, extract_words
from collections import defaultdict, OrderedDict
import pickle
import cPickle as pickle
import time 

class DocumentStreamReader:
    def __init__(self, paths):
        self.paths = paths

    def open_single(self, path):
        return gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb')

    def __iter__(self):
        for path in self.paths:
            with self.open_single(path) as stream:
                while True:
                    sb = stream.read(4)
                    if sb == '':
                        break

                    size = struct.unpack('i', sb)[0]
                    msg = stream.read(size)
                    doc = document_pb2.document()
                    doc.ParseFromString(msg)
                    yield doc


def parse_command_line():
    parser = argparse.ArgumentParser(description='compressed documents reader')
    parser.add_argument('coding', nargs=1, help='Coding', type=str)
    parser.add_argument('files', nargs='+', help='Input files (.gz or plain) to process')
    return parser.parse_args()

def varbyte(x):
    def code(s):
        res = bytearray()
        while(s >= 128):
            res.extend([s & 127])
            s >>= 7
        res.extend([(s & 127) + 128])
        return res
    res = bytearray()
    for s in x:
        res.extend(code(s))
    return res

def varbyte_decode(x):
    it = iter(x)
    k = 0
    x = 0
    res = []
    for s in it:
        if s < 128:
            x += (1 << 7 * k) * s
            k += 1
        elif s <= 255:
            x += (1 << 7 * k) * (s - 128)
            res.append(x)
            k = 0
            x = 0
        else:
            raise RuntimeError
    return res

simple9_ind = [1,2,3,4,5,7,9,14,28,100]

def simple9(x):
    global simple9_ind
    def pack(x, i):
        '''
        i - count of numbers
        k - len of number
        '''
        k = int(28 / simple9_ind[i])
        z = 28 - k * simple9_ind[i]
        
        #for p in range(simple9_ind[i]):
        #    assert((x[p] >> k) == 0)

        cur = 32
        payload = 0
        
        cur -= 4
        payload += (i << cur)
        cur -= z

        for l in range(simple9_ind[i]):
            cur -= k
            payload += (x[l] << cur) 

        mask = (1 << 8) - 1
        return bytearray([(payload >> 24) & mask, (payload >> 16) & mask, (payload >> 8) & mask, (payload >> 0) & mask]) 
   
    def packit(x, n):
        k = x[0].bit_length()
        i = 0
        while(simple9_ind[i] * k <= 28 and simple9_ind[i] <= n and i < 9):
            i += 1
            if simple9_ind[i] > n:
                break

            if i > 4:
                for j in range(simple9_ind[i-1], simple9_ind[i]):
                    k = max(k, x[j].bit_length())
            else:
                k = max(k, x[simple9_ind[i]-1].bit_length())
        i -= 1
        return pack(x, i), simple9_ind[i]

    res = bytearray()
    n = len(x)
    while(n > 0):
        q, i = packit(x, n)
        x = x[i:]
        n -= i
        res.extend(q)
    return res

def simple9_decode(x):
    def unpack(x):
        i = simple9_ind[(x[0] >> 4)]
        k = int(28 / i)
        z = 28 - i * k
        mask  = (1 << 4) - 1
        payload = ((x[0] & mask) << 24) + (x[1] << 16) + (x[2] << 8) + (x[3] << 0)
        res = [0] * i
        for j in range(28 - z):
            l = int(j / k)
            bit = (payload >> j) % 2
            res[i - l - 1] += (bit << (j - l * k))

        return res

    it = list(x)
    res = []
    for i in range(0, len(it), 4):
        res.extend(unpack(it[i:i+4]))
    return res
        


class CoderFile:
    def __init__(self, f, g, coding=lambda x : x):
        self.f = f
        self.g = g
        self.coding = coding
        self.last_hash = None
        self.last_diff = 0

    def write(self, key, lst):

        header = list(map(ord, key))
        header.append(ord('\n'))
        header.append(len(lst))
        code = self.coding(header + lst)

        h = hash(key)
        if self.last_hash == None:
            self.last_hash = h
            self.g.write(struct.pack('q', self.last_hash))
            self.g.write(struct.pack('i', self.last_diff))
            '''
            self.g.write(str(self.last_hash) + ':' + str(self.last_diff) + ':')
            '''

        if self.last_hash < h:
            diff = self.f.tell()
            self.last_hash = h
            self.g.write(struct.pack('i', diff - self.last_diff))
            self.g.write(struct.pack('q', self.last_hash))
            self.g.write(struct.pack('i', diff))
            '''
            self.g.write(str(diff - self.last_diff) + '\n')
            self.g.write(str(self.last_hash) + ':' + str(diff) + ':')
            '''
            self.last_diff = diff
        elif self.last_hash > h:
            raise RuntimeError
        try:
            self.f.write(code)
        except:
            print(code)
        
    def close(self):
        self.f.close()
        self.g.write(struct.pack('i', -1))
        self.g.close()

def Pack(d, docids, coding):
    pickle.dump(docids, open('./docids.txt', 'w'))
    with open('./index.txt', 'w') as f, open('./dict.txt', 'w') as g:
        f = CoderFile(f, g, coding)
        lst = sorted(d.items(), key=lambda t : hash(t[0]))
        for i in range(len(lst)):
            f.write(lst[i][0], lst[i][1])
        f.g.write(struct.pack('i', -1))

if __name__ == '__main__':
    files = parse_command_line().files
    coding = parse_command_line().coding[0]
    if coding == 'varbyte':
        coding = varbyte
        encoding = varbyte_decode
        pickle.dump('varbyte', open('coding_type.txt', 'w'))
    elif coding == 'simple9':
        ''' he is working very slowly, so i cannot pass 25, using simple9 (not searching, just indexing)
        coding = simple9
        encoding = simple9_decode
        pickle.dump('simple9', open('coding_type.txt', 'w'))
        #'''
        coding = varbyte
        encoding = varbyte_decode
        pickle.dump('varbyte', open('coding_type.txt', 'w'))
        #'''
    else:
        raise NotImplemented

    reader = DocumentStreamReader(files)
    d = defaultdict(list)
    docid = -1
    docids = dict()
    for doc in reader:
        docid += 1
        words = extract_words(doc.text)
        for word in words:
            d[word].append(docid)
        docids[docid] = doc.url
    Pack(d, docids, coding)
    
    '''
    for key in docids.keys():
        print(key)
        print(docids[key])
    #'''
