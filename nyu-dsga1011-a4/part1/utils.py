import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Synonym replacement: each word longer than 3 letters is replaced, with
    # probability 0.6, by a WordNet synonym from its most common sense.
    def get_synonym(word):
        if word[0].isupper():                          # skip names like "Hanks"
            return word
        candidates = []
        for syn in wordnet.synsets(word)[:1]:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():       # "Movie" -> "movie" isn't a synonym
                    candidates.append(name)
        if len(candidates) == 0:
            return word
        return random.choice(candidates)

    words = word_tokenize(example["text"])
    new_words = []
    for w in words:
        if random.random() < 0.6 and len(w) > 3:
            new_words.append(get_synonym(w))
        else:
            new_words.append(w)
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
