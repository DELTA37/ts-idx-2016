import re
from collections import defaultdict
SPLIT_RGX = re.compile(r'\w+', re.U)

def extract_words_doc(text, doc):
    words = re.findall(SPLIT_RGX, text)
    return defaultdict(list, map(lambda s: (s.lower(), doc), words))
    
def extract_words(text):
    words = re.findall(SPLIT_RGX, text)
    return set(map(lambda s: s.lower(), words))
    
