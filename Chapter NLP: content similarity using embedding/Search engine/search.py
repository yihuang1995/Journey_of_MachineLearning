# Launch with
#
# gunicorn -D --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import sys
from related_word_script import *
from ranking import * 
import pandas as pd
from collections import defaultdict
from preprocessing_ngram import *
from ngram import *

app = Flask(__name__, static_folder='static', static_url_path='/static')
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == "POST" and request.form["submit"] == 'Search':
        q = request.form["nm"]
        season = request.form["season"]
        if not q or len(q) > 20: # or len(q.split()) != 1
            return render_template('search.html')
        return redirect(url_for("articles", word=q,season=season))
    elif request.method == "POST" and request.form["submit"] == 'Phrase Search':
        q = request.form["nm"]
        season = request.form["season"]
        return redirect(url_for("ngram", word=q,season=season))
    return render_template('search.html')

@app.route("/search/ngram/<word>/<season>/ngram")
def ngram(word,season):
    query_df = pd.read_csv(query_df_path)
    new_df = query_df[query_df['quarter']==f'{season.upper()}']
    search_vec_list = ngram_word2vec(word,model_ngram)
    result_dict = ngram_matching(scripts,search_vec_list,model_ngram)
    result_dict = { stories[key]: result_dict[key] for key in result_dict} # filter only particular season
    # result_dict['DEMI_LOVATO_NEW'] = result_dict.pop('DEMI_LOVATO_NEW_2') # due to scrape issue
    # result_dict['Melody_of_Your_Heartbeat'] = result_dict.pop('Melody_of_Your_Heartbeat_2') # due to scrape issue
    result_dict = { key: result_dict[key] for key in result_dict if key in set(new_df.story.values)} # filter only particular season
    related_result = list(result_dict.items())[:100] # top 100 related stories
    #story_info_dict = new_df.set_index('story')['score'].to_dict()
    return render_template('articles_ngram.html', word=word, stories_info=related_result)


@app.route("/search/<word>/<season>")
def articles(word,season):
    """Show a list of article titles"""
    word_similarity = get_related_words(model, word.split()) # get top related word to the search word
    word_count_per_story_file = Path('chunk_count_per_story.json')
    with open(word_count_per_story_file) as f:
        word_count_per_story = json.loads(f.read())
    word_count_per_story_season, stories_info_dict = get_story(season, word_count_per_story) 
    # drop purchased story in dictionary
    word_count_per_story_season = {k: v for k, v in word_count_per_story_season.items() if k not in purchased_list}
    stories_info_dict = {k: v for k, v in stories_info_dict.items() if k not in purchased_list}


    stories_target_count, stories_similar_count = get_target_n_similar_stories(word_similarity, word_count_per_story_season)
    # related_stories_top5 = list(stories_avg_similarity.keys())
    #ranking
    word_count_dict = merge_cnt(stories_target_count,stories_similar_count) #ttl cnt for word and similar word
    df = merge_dataframe(query_df_path,word_count_dict,season) #add word count to df
    story_with_score = score(df, feature_list) #df with story and rank score
    # drop purchased row
    story_with_score = story_with_score[~story_with_score['story'].isin(purchased_list)]

    result_csv = story_with_score[story_with_score['word_count']>0]
    result_csv.to_csv('result.csv')
    sorted_story = story_with_score.set_index('story')['score'].to_dict()
    stories_target_count = dict(sorted(stories_target_count.items(), key=lambda item: sorted_story[item[0]],reverse = True))
    stories_similar_count = dict(sorted(stories_similar_count.items(), key=lambda item: sorted_story[item[0]],reverse = True))
    print(stories_target_count)
    return render_template('articles.html', word=word,
                           target_articles=stories_target_count, similar_articles=stories_similar_count,
                           target_len=len(stories_target_count), similar_len=len(stories_similar_count),
                           stories_info=stories_info_dict)

@app.route('/download')
def download_file():
    result_df = 'result.csv'
    return send_file(result_df,as_attachment = True, cache_timeout=0)

# initialization
model = KeyedVectors.load('glove.kv')
model_ngram = KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin.gz', binary=True)
df, stories, scripts = load_script_ngram('../scrape/scraping/ugc2020-2021')
# word_similarity = get_related_words(model,'lgbt')
#
# word_count_per_story_file = Path('word_count_per_story.json')
# if word_count_per_story_file.is_file():
#     with open(word_count_per_story_file) as f:
#         word_count_per_story = json.loads(f.read())
# else:
#     ugc_path = '../ugc2020q4'
#     _, stories, scripts = load_script(ugc_path)  # stories, scripts are lists
#     _, word_count_per_story = get_word_count(stories, scripts)  # word_count_per_story is nested dictionary
#
# word_count_per_story['DEMI_LOVATO_NEW'] = word_count_per_story['DEMI_LOVATO_NEW_2']
# word_count_per_story['Melody_of_Your_Heartbeat'] = word_count_per_story['Melody_of_Your_Heartbeat_2']

## for ranking
feature_list = ['word_count', 'num_reads', 'chapter5_retention_rate', 'gem_spent', 'gem_per_read']
query_df_path = '../scrape/story_info.csv'

## purchased_story
purchased = pd.read_csv('../purchased/Editorial Story Discovery - Purchase List.csv')
purchased_list = purchased[purchased['Status']=='Complete']['UGC Story ID'].unique()

def merge_cnt(word_dict,similar_word_dict):
    word_cnt = defaultdict(int)
    for item in word_dict.keys():
        word_cnt[item] = sum(word_dict[item].values())
    for item in similar_word_dict.keys():
        word_cnt_list = similar_word_dict[item].values()
        word_cnt[item] = word_cnt.get(item,0) + sum([item for item in similar_word_dict[item].values()])
    return word_cnt

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
