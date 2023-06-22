import pandas as pd
import numpy as np
import datetime
from IPython.display import HTML, display

import re
import string
import os
import codecs
import spacy
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Common male names
# credit： https://www.cs.cmu.edu/Groups/AI/areas/nlp/corpora/names/male.txt
male_path = '../words/male.txt'
with open(male_path) as f:
    male_names = f.readlines()
male_names = [n.rstrip('\n').lower() for n in male_names[6:]]

# Common female names
female_path = '../words/female.txt'
with open(female_path) as f:
    female_names = f.readlines()
female_names = [n.rstrip('\n').lower() for n in female_names[6:]]

stop_words_path = '../words/stop_words.txt'
with open(stop_words_path) as f:
    stop_words = f.readlines()
stop_words = [n.rstrip('\n') for n in stop_words[2:]]

nouns = pd.read_csv('../words/most-common-nouns-english.csv')
common_nouns = nouns['Word'].tolist()

setting_words = ["colors", "shapes", "hairstyles", "tones"]

common_words = set(male_names + female_names + common_nouns + stop_words)


def filelist(root):
    """Return a fully-qualified list of filenames under root directory"""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return allfiles


def get_text(filename):
    """
    Load and return the text of a text file.
    Use codecs.open() function not open().
    """
    f = codecs.open(filename, mode='r', encoding='utf-8', errors='ignore')
    lines = f.readlines()

    dialogue = []
    for line in lines:
        if line.startswith('    ') and not line.startswith('        '):

            for setting_word in setting_words:
                if setting_word in line:
                    break
            else:
                dialogue.append("".join(line.split('|')[0::2]))

    return "".join(dialogue)


def process_ngram(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    """
    #text = text.lower()
    # text = re.compile(text)
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    words_list = text.split(' ')
    words_list = [w for w in words_list if len(w) > 2]  # ignore a, an, to, at, be, ...
    words_list = [w for w in words_list if w not in common_words]
    processed_text = " ".join(words_list)
    return processed_text


def load_script_ngram(dirname):
    """
    load all scripts(episode stories), return a dataframe with story 
    and corresponding text, stories name (list), story text (list)
    """
    scripts_path = filelist(dirname)
    stories = []
    texts = []
    for path in scripts_path:
        story = path.split('/')[-1][:-4]
        text = get_text(path)
        stories.append(story)
        texts.append(process_ngram(text))
    df = pd.DataFrame({'story': stories, 'script': texts})

    stories = df['story'].tolist()
    scripts = df['script'].tolist()
    return df, stories, scripts





if __name__ == "__main__":
    ugc_path = '../scraping_2020/ugc2020'

    df_2020_q4, stories, scripts = load_script(ugc_path)

