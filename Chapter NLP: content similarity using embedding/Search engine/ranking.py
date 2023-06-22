#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import string


def merge_dataframe(query_df_path,story_word_count,season):
    """
    merge query story information results and the word count dataframe for particular season
    """
    query_df = pd.read_csv(query_df_path)
    new_df = query_df[query_df['quarter']==f'{season.upper()}'].copy() #filter data in particular season 
    new_df['word_count'] = new_df['story'].map(story_word_count)
    new_df['word_count'].fillna(0,inplace = True)
    return new_df


def score(mydf, cols):
    """
    Ranking algorithm with Coefficient of Variance Method. We first nomalize all the feature columns,
    then calculate the final score from these nomalized features
    https://en.wikipedia.org/wiki/Coefficient_of_variation
    """
    df = mydf.copy()
    df = df[df['word_count']>0]    
    v_list = []
    normalized_col = []
    for col in cols:
        mean = df[col].mean()
        std = np.std(df[col])
        df[f'{col}_nomalized'] = (df[col]-mean)/std
        normalized_col.append(f'{col}_nomalized')
        v_list.append(std/mean)
    weight_list = [v/np.array(v_list).sum() for v in v_list]
    weight_dict = {normalized_col[i]:weight_list[i] for i in range(len(cols))} #get a weight dict for each column
    
    df_score = df[normalized_col] * pd.Series(weight_dict) #calculate score for each column
    df['score'] = df_score.sum(axis=1) #Add all column score to get the final score
    
        
    return df
