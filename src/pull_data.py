
# %%
from pybaseball import  playerid_lookup # type: ignore
from pybaseball import  statcast_pitcher # type: ignore
from pybaseball import  statcast # type: ignore
import pandas as pd
import numpy as np

def pull_data (first_name, last_name, team_abbreviation):
    name = playerid_lookup(last_name, first_name)
    
    if len(name) > 1:
        print()
    df = statcast_pitcher('2023-04-01', '2023-09-30', name['key_mlbam'])

    path_1 = '/Users/cstone/Documents/Projects/MLB Pitch Prediction/data/raw'
    return df


def remove_factors (df):
    '''
    This identifies all columns with missing values, except for on_base columns. 
    It also further removes all columns that would be insignigicant to predicting pitch type
    '''
    #missing values
    missing = df.isna().sum().reset_index()
    missing = missing.loc[missing[0] > 0]
    removal_list = missing['index'].to_list()

    #keeping on_base columns and changing to binary data types
    on_base_cols = ['on_1b', 'on_2b', 'on_3b']
    for i in on_base_cols:
        df[i] = df[i].fillna(0)
        df[i] = np.where(df[i]>1, 1, df[i])

    #initializing removal list
    removal_list = [x for x in removal_list if x not in on_base_cols]
    removal_list+= ['release_speed', 'release_pos_x',
                'release_pos_z', 'player_name', 'batter', 'pitcher', 'description',
                'zone', 'des', 'game_type', 'home_team',
                'away_team', 'type', 'game_year', 'pfx_x', 'pfx_z',
                'plate_x', 'plate_z', 'fielder_2', 'vx0', 'vy0', 'vz0', 'ax', 'ay',
                'az', 'sz_top', 'sz_bot', 'effective_speed', 'release_extension',
                'game_pk', 'pitcher.1', 'fielder_2.1', 'fielder_3', 'fielder_4',
                'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9',
                'release_pos_y', 'pitch_name', 'game_date','post_away_score',
                'post_home_score', 'post_bat_score', 'post_fld_score',
                ]
    
    #dropping removal list from dataframe
    df2 = df.drop(removal_list, axis=1)
    return df2
    
def transform_data(df):

    '''
    This transforms all remaaining columns with categorical data types to either binary or dummies
    '''
    df['batter_is_right'] = np.where(df['stand'] == 'R', 1, 0)
    df['pitcher_is_right'] = np.where(df['p_throws'] == 'R', 1, 0)
    df['inning_top'] = np.where(df['inning_topbot'] == 'Top', 1, 0)
    df = df.drop(['stand', 'p_throws', 'inning_topbot'], axis=1)
    df = pd.get_dummies(df, columns=['if_fielding_alignment', 'of_fielding_alignment'], drop_first=True)
    
    return df

def return_cleaned_df (df, path):
    #transforming fielding alignment
    transformed_data = pd.get_dummies(df, columns=['if_fielding_alignment', 'of_fielding_alignment'], drop_first=True)

    path = '/Users/cstone/Documents/Projects/MLB Pitch Prediction/data/raw/cleaned'
    pd.to_csv()


