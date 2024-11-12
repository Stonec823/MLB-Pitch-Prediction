from pybaseball import  playerid_lookup # type: ignore
from pybaseball import  statcast_pitcher # type: ignore
from pybaseball import  statcast # type: ignore
import pandas as pd
import numpy as np

def pull_data (first_name, last_name, start_date, end_date, team_abbreviation = None):
    
    '''
    This makes the call to the pybaseball function player_id lookup to obtain the id value for the pitcher
    
    If it detects multiple pitchers of the same name, if will select the most recent pitcher to debut
    
    Then it will use the id to obtain their stats in a given timeframe. In this case for the 2023 season
    '''
    name = playerid_lookup(last_name, first_name)
    if len(name) > 1:
        name = name.sort_values(by='mlb_played_first', ascending=False)
        name = name.head(1)
    id = name['key_mlbam'].values[0]
    df = statcast_pitcher(start_date, end_date, id)
    return df

def add_lag_features (df):
    
    '''
    This take several steps to add lag features and other aggregate features that should be helpful in moedling
    
    '''
    pitch_dict = {'FF':0, 'FA':0,
              'FT':1, 'SI':1,
              'FC':2,
              'CU':3,'KC':3,'CS':3,'EP':3,
              'SL':4,
              'CH':5,'FS':5,'FO':5,'SC':5,
              'KN':6,
              'PO':np.nan}

    # Map old pitch types to new mapping
    df['pitch_type_map'] = df['pitch_type'].map(pitch_dict)
    df.dropna(subset=['pitch_type_map'], inplace = True)
    
    #add lag pitches
    df['prev_pitch_1'] = df.groupby('pa_id')['pitch_type_map'].shift(1)
    df['prev_pitch_2'] = df.groupby('pa_id')['pitch_type_map'].shift(2)
    df['prev_pitch_3'] = df.groupby('pa_id')['pitch_type_map'].shift(3)
    
    
    pitch_types = df['pitch_type'].unique()

    # Function to calculate trailing 5-game proportions
    def calculate_trailing_proportions(row):
        current_game = row['game_pk']
        
        
        recent_games = df[df['game_pk'] < current_game]['game_pk'].drop_duplicates().tail(5)
        
       
        recent_pitches = df[df['game_pk'].isin(recent_games)]
        
       
        proportions = {}
        total_pitches = len(recent_pitches)
        for pitch in pitch_types:
            count = recent_pitches[recent_pitches['pitch_type'] == pitch].shape[0]
            proportions[f'prop_{pitch}'] = count / total_pitches if total_pitches > 0 else 0
        
        return pd.Series(proportions)

    # Apply the function row-wise to calculate trailing 5-game proportions
    proportion_cols = df.apply(calculate_trailing_proportions, axis=1)
    df = pd.concat([df, proportion_cols], axis=1)
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
   
    removal_list =  ['release_speed', 'release_pos_x',
            'release_pos_z', 'player_name', 'batter', 'pitcher', 'description',
            'zone', 'des', 'game_type', 'home_team',
            'away_team', 'type', 'game_year', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'fielder_2', 'vx0', 'vy0', 'vz0', 'ax', 'ay',
            'az', 'sz_top', 'sz_bot', 'effective_speed', 'release_extension',
            'game_pk', 'pitcher.1', 'fielder_2.1', 'fielder_3', 'fielder_4',
            'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9',
            'release_pos_y', 'pitch_name', 'game_date','post_away_score',
            'post_home_score', 'post_bat_score', 'post_fld_score','on_3b', 'on_2b','on_1b'
            ,'hit_location', 'bb_type', 'events', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'woba_value', 'woba_denom', 'babip_value', 'iso_value'
            ,'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id'
            ,'hit_distance_sc', 'launch_speed', 'launch_angle', 'release_spin_rate','hc_x', 'hc_y'
            ]
    #dropping removal list from dataframe
    df2 = df.drop(removal_list, axis=1)
    return df2
    
def encode(df):

    '''
    This transforms all remaaining columns with categorical data types to either binary or dummie variables
    '''
    df['batter_is_right'] = np.where(df['stand'] == 'R', 1, 0)
    df['pitcher_is_right'] = np.where(df['p_throws'] == 'R', 1, 0)
    df['inning_top'] = np.where(df['inning_topbot'] == 'Top', 1, 0)
    df = df.drop(['stand', 'p_throws', 'inning_topbot'], axis=1)
    df = pd.get_dummies(df, columns=['if_fielding_alignment', 'of_fielding_alignment'], drop_first=True)
    
    return df
        
        
def pull_pitcher_data(first_name, last_name, start_date, end_date):
    
    df = pull_data(first_name, last_name, start_date, end_date)
    df = add_lag_features(df)
    df = remove_factors(df)
    df = encode(df)
    
    return df