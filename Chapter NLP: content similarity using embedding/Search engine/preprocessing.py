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
        if line.startswith('    ') and not line.startswith('        '): #extract useful text from story file

            for setting_word in setting_words:
                if setting_word in line:
                    break
            else:
                dialogue.append("".join(line.split('|')[0::2]))

    return "".join(dialogue)


def process(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    """
    text = text.lower()
    # text = re.compile(text)
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    words_list = text.split(' ')
    words_list = [w for w in words_list if len(w) > 2]  # ignore a, an, to, at, be, ...
    words_list = [w for w in words_list if w not in common_words]
    processed_text = " ".join(words_list)
    return processed_text


def load_script(dirname):
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
        texts.append(process(text))
    df = pd.DataFrame({'story': stories, 'script': texts})

    stories = df['story'].tolist()
    scripts = df['script'].tolist()
    return df, stories, scripts


def get_word_count(stories, scripts):
    """
    for each story, record word count
    """
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(scripts)
    word_count_matrix = pd.DataFrame(data=X.toarray(), columns=vectorizer.get_feature_names())

    word_counts = []
    for i in range(len(word_count_matrix)):
        high_count_scores = word_count_matrix.iloc[i, :].sort_values(ascending=False)
        high_count_scores_wo_zero = high_count_scores[high_count_scores > 0].to_dict()
        word_counts.append(high_count_scores_wo_zero)

    return word_count_matrix, dict(zip(stories, word_counts))


def calc_tfidf(stories, scripts):
    """
    calculate stories' tfidf and score, this function is not used in the final search 
    engine
    """

    X, _ = get_word_count(stories, scripts)
    trans = TfidfTransformer()
    D = trans.fit_transform(X).toarray()
    output = pd.DataFrame(data=D, columns=X.columns)

    script_list_tfidf = []

    for i in range(len(output)):
        top = 10
        high_tfidf_scores = output.iloc[i, :].sort_values(ascending=False)
        high_tfidf_scores_zero = high_tfidf_scores[high_tfidf_scores > 0]
        high_tfidf_words = high_tfidf_scores_zero.index.tolist()
        script_list_tfidf.append(high_tfidf_words)

    return dict(zip(stories, script_list_tfidf))

"""
After runing this file, a story - word count json file will be generated for future use
"""

if __name__ == "__main__":
    ugc_path = '../scrape/scraping/ugc2020-2021'

    df, stories, scripts = load_script(ugc_path)
    #print(df.shape)
    word_count_matrix, word_counts = get_word_count(stories, scripts)
    #print(word_counts)
    json = json.dumps(word_counts)
    f = open("word_count_per_story.json","w")
    f.write(json)
    f.close()
