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
    try: 
        name = playerid_lookup(last_name, first_name)
        if len(name) > 1:
            name = name.sort_values(by='mlb_played_first', ascending=False)
            name = name.head(1)
        id = name['key_mlbam'].values[0]
        df = statcast_pitcher(start_date, end_date, id)
        return df
    except:
        print('Please check name spelling, or add team name abbreviation for more specificity')

def add_lag_features (df):
    df = df.copy()
    '''
    This takes several steps to add lag features and other aggregate features that should be helpful in modeling
    
    '''
    pitch_dict = {'FF':0, 'FA':0,
              'FT':1, 'SI':1,
              'FC':2,
              'CU':3,'KC':3,'CS':3,'EP':3,
              'SL':4, 'ST': 4,
              'CH':5,'FS':5,'FO':5,'SC':5,
              'KN':6,
              'PO':np.nan}

    # Map old pitch types to new mapping
    df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
    # df['pa_id'] = str(df['game_pk']) + "-" + str(df['at_bat_number'])
    # df['pitch_id'] = str(df['pa_id']) + "-" + str(df['pitch_number'])
    df['pitch_type_map'] = df['pitch_type'].map(pitch_dict)
    df['is_fastball'] = np.where(df['pitch_type_map'] == 0, 1, 0)
    #dropping rows with Nan
    df = df.dropna(subset=['pitch_type_map'])
    df['pitch_type_map'] = df['pitch_type_map'].astype(int)
    df.dropna(subset=['pitch_type_map'], inplace = True)

   
    #add lag pitches
    df['prev_pitch_1'] = df['pitch_type_map'].shift(1)
    df['prev_pitch_2'] = df['pitch_type_map'].shift(2)
    df['prev_pitch_3'] = df['pitch_type_map'].shift(3)
    
    
    pitch_types = df['pitch_type'].unique()
    return df
    # Function to calculate trailing 5-game proportions
    # def calculate_trailing_proportions(row):
    #     current_game = row['game_pk']
        
        
    #     recent_games = df[df['game_pk'] < current_game]['game_pk'].drop_duplicates().tail(5)
        
       
    #     recent_pitches = df[df['game_pk'].isin(recent_games)]
        
       
    #     proportions = {}
    #     total_pitches = len(recent_pitches)
    #     for pitch in pitch_types:
    #         count = recent_pitches[recent_pitches['pitch_type'] == pitch].shape[0]
    #         proportions[f'prop_{pitch}'] = count / total_pitches if total_pitches > 0 else 0
        
    #     return pd.Series(proportions)

    # # Apply the function row-wise to calculate trailing 5-game proportions
    # proportion_cols = df.apply(calculate_trailing_proportions, axis=1)
    # df = pd.concat([df, proportion_cols], axis=1)
    # return df

def remove_factors (df):
    
    df = df.copy()
    '''
    This will remove columns with missing values, and also remove lots of features that occur during or after the pitch 
    since we will only be interested in predicitive and situational features
    
    '''
    #missing values
    missing = df.isna().sum().reset_index()
    missing = missing.loc[missing[0] > 0]
    removal_list = missing['index'].to_list()

    
   
    # removal_list = ['release_speed', 'release_pos_x',
    #         'release_pos_z', 'player_name', 'batter', 'pitcher', 'description',
    #         'zone', 'des', 'game_type', 'home_team',
    #         'away_team', 'type', 'game_year', 'pfx_x', 'pfx_z',
    #         'plate_x', 'plate_z', 'fielder_2', 'vx0', 'vy0', 'vz0', 'ax', 'ay',
    #         'az', 'sz_top', 'sz_bot', 'effective_speed', 'release_extension', 'fielder_3', 'fielder_4',
    #         'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9',
    #         'release_pos_y', 'pitch_name', 'game_date','post_away_score',
    #         'post_home_score', 'post_bat_score', 'post_fld_score','on_3b', 'on_2b','on_1b',
    #         'hit_location', 'bb_type', 'events', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'woba_value', 'woba_denom', 'babip_value', 'iso_value',
    #         'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id',
    #         'hit_distance_sc', 'launch_speed', 'launch_angle', 'release_spin_rate','hc_x', 'hc_y',
    #         'pitcher_days_since_prev_game', 'batter_days_since_prev_game',
    #         'pitcher_days_until_next_game', 'batter_days_until_next_game',
    #         'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in',
    #         'arm_angle', 'delta_run_exp', 'bat_speed', 'swing_length', 'estimated_ba_using_speedangle',
    #    'estimated_woba_using_speedangle', 'launch_speed_angle', 'estimated_slg_using_speedangle', 'spin_axis',
    #    'delta_pitcher_run_exp']
    #dropping removal list from dataframe
    df = df.drop(removal_list, axis=1)
    return df
    
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
    # df = remove_factors(df)
    df = encode(df)
    
    #filtering to a final featureset
    
    final_features = ['game_pk', 'at_bat_number', 'pitch_number','pitch_type' ,'pitch_type_map',
        'is_fastball', 'balls', 'strikes', 'outs_when_up', 'inning',
       'home_score', 'away_score',
       'bat_score', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat',
       'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3',
       'batter_is_right', 'pitcher_is_right', 'inning_top'
       ]
    #prop_columns = [col for col in df.columns if col.startswith('prop')]
    
    #+ prop_columns
    # this is dropping any NA rows, or 0 rows
    df = df[final_features]
    #df = df.sort_values(by=['game_pk', 'at_bat_number', 'pitch_number'])
    #df = df.dropna(axis='columns')
    

    # Drop rows where any 'prop' column equals 0
    #df = df[~(df[prop_columns] == 0).any(axis=1)]
    
    return df