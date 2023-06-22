import sys
import re
import string
import os
import numpy as np
import codecs
import json
import pandas as pd
from pathlib import Path
from gensim.models import KeyedVectors
from preprocessing import load_script, get_word_count, calc_tfidf
from sklearn.feature_extraction.text import CountVectorizer
from helper import Timer
from collections import defaultdict


def get_related_words(model, word: list, n=5, cnt=2000000):
    """
    get top n related word list 
    """
    word_list = model.most_similar(word, topn=n)  # or word.lower()
    # word_list = [(word, corr) for word, corr in word_list if model.vocab[word].count > cnt]
    word_list.insert(0, (' '.join(word), 1))

    # filter out the words with length less than 2
    word_list = [word for word in word_list if len(word[0]) > 2]

    return word_list


def get_target_n_similar_stories(word_similarity: list, word_count_per_story: dict):
    """
    get stories with target words and similar words
    """
    target_word_count = {}
    similar_word_count = {}

    stories_target_count = {}
    stories_similar_count = {}
    stories_avg_similarity = {}

    for story, word_count in word_count_per_story.items():
        # script with target word
        target_word_count = 0
        similar_word_count = defaultdict(int)

        for word, count in word_count.items():
            if word_similarity[0][0] in word:
                target_word_count += count

            for similar_word, similarity in word_similarity[1:]:
                if similar_word in word:
                    similar_word_count[similar_word] += count

        if target_word_count != 0:
            stories_target_count[story] = {word_similarity[0][0]: target_word_count}

        if similar_word_count != {}:
            stories_similar_count[story] = similar_word_count

    return stories_target_count, stories_similar_count


def get_story(season, word_count_per_story):
    # story_data = pd.read_csv(f'../scraping_2020/2020{season}.csv')
    with open(f'../scrape/df_json/{season}.json') as f:
        json_file = f.read()
        stories_info = json.loads(json_file)

    word_count_per_story_season = {}
    stories_info_dict = {}
    for i in stories_info:
        word_count_per_story_season[i['story']] = word_count_per_story.get(i['story'], {})
        stories_info_dict[i['story']] = list(i.values())[1:]
    return word_count_per_story_season, stories_info_dict


if __name__ == "__main__":
    timer = Timer()
    # model = KeyedVectors.load_word2vec_format('../w2v_dataset/GoogleNews-vectors-negative300.bin', binary=True)
    # model = KeyedVectors.load_word2vec_format('../w2v_dataset/glove.6B.50d.w2vformat.txt', binary=False)
    # model.save("glove.kv")
    model = KeyedVectors.load('glove.kv')

    # download link: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
    # google word2vec link: https://code.google.com/archive/p/word2vec/
    # glove link: https://nlp.stanford.edu/projects/glove/
    # convert glove to w3v: python -m gensim.scripts.glove2word2vec --input glove.6B.50d.txt --output glove.6B.50d.w2vformat.txt

    word_similarity = get_related_words(model, 'lgbt')
    print(f"running time of model is {timer.stop():.5f} sec")

    word_count_per_story_file = Path('word_count_per_story.json')
    if word_count_per_story_file.is_file():
        with open(word_count_per_story_file) as f:
            word_count_per_story = json.loads(f.read())
    else:
        ugc_path = '../scrap/scraping/ugc2020-2021'
        _, stories, scripts = load_script(ugc_path)  # stories, scripts are lists
        _, word_count_per_story = get_word_count(stories, scripts)  # word_count_per_story is nested dictionary

    stories_target_count, stories_similar_count, stories_avg_similarity = get_target_n_similar_stories(word_similarity,
                                                                                                       word_count_per_story)

    # print(f"there are {len(stories_target_count)} stories that contain the input word directly")
    # print(f"there are {len(stories_similar_count)} stories that contain the similar word with input")
    # print(f"total running time is {timer.stop():.5f} sec")
    # print(f"{list(stories_avg_similarity.keys())[:5]}")
    print(stories_target_count)
