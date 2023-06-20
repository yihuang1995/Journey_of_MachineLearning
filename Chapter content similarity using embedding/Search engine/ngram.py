import numpy as np
from preprocessing import load_script
from gensim.models import KeyedVectors


def ngram_word2vec(words,model):
    """
    transfer searching words into vectors that is in the model's dict, 
    the model could be GloVe, Word2Vec..
    """
    vector_list = []
    word_list = words.split()
    for word in word_list:
        try:
            vec = model[word]
        except:
            print("Word not in dictionary") 
            return 
        vector_list.append(vec)
    return vector_list

def article_min_distance(article,vector_list,model):
    """
    For a particular article, use a window to scan each word pairs in the article,
    calculate the distance with search words' vector, return the minimal distance and 
    corresponding words
    """
    window_size = len(vector_list)
    word_list = article.split()
    min_distance = float('inf')
    words = ''
    for i in range(len(word_list)-window_size + 1):
        distance = 0
        try:
            for j in range(window_size):
                vec = model[word_list[i+j]]
    #                 distance += np.dot(vector_list[j],vec)
                distance +=  np.linalg.norm(vector_list[j]-vec)
#             print(distance)
#             print(' '.join(word_list[i:i+window_size]))
            if distance < min_distance:
                min_distance = distance
                words = ' '.join(word_list[i:i+window_size])
        except:
            #print(f'Error: {" ".join(word_list[i:i+window_size])}')
            continue
    return min_distance, words

def ngram_matching(scripts,search_vec_list,model):
    """
    Main function for Ngram searching, given story scripts, return each story's 
    minumal distance with the searching words and show what words is same/similar to
    the searching words 
    """
    distance_list = []
    related_list = []
    for item in scripts:
        min_distance, words = article_min_distance(item,search_vec_list,model)
        distance_list.append(min_distance)
        related_list.append(words)
    result_dict = {i:(related_list[i],distance_list[i]) for i in range(len(related_list))}
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1][1],reverse=False))
    return result_dict