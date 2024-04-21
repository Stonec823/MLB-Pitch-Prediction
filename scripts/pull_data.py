
# %%
from pybaseball import  playerid_lookup
from pybaseball import  statcast_pitcher
from pybaseball import  statcast
import pandas as pd
import numpy as np


# %%
#This function finds id info for any player in baseball history. I'll choose one of the more popular and successful pitchers in recent years Blake Snell for my first example
# a = playerid_lookup('snell', 'blake')

# There was a Blake snell that played in the early 1900's, so just filtering odwn. We'll extract the id to use for stat searching
# a[a['mlb_played_first'] > 2000.0]

# %%
#This data is on a pitch level. This is df contains all 3000+ pitches thrown by Snell in the designated timeframe.

df = statcast_pitcher('2023-04-01', '2023-09-30', 605483)

path_1 = '/Users/cstone/Documents/Projects/MLB Game Prediction/data/raw'
# This table contains some factors that I assume will be key in predicting pitch type. 

# %%
#assessing missing values
missing = df.isna().sum().reset_index()
missing = missing.loc[missing[0] > 0]



# %%
#initializg a list of null factors to possibly be trimmed 
missing_list = missing['index'].to_list()


#These conatin player id's for whoever is on base. I'll replace with binary variables

on_base_cols = ['on_1b', 'on_2b', 'on_3b']

#converting on base to binary variables
for i in on_base_cols:
    df[i] = df[i].fillna(0)
    df[i] = np.where(df[i]>1, 1, df[i])



# %%
#removing on base columns from removal list
missing_list_without_ob = [x for x in missing_list if x not in on_base_cols]

#new dataframe with null columns removed, while keeping on_base columns
df2 = df.drop(missing_list_without_ob, axis=1)


# %%

#initializing list of factors that can be safely removed
missing_list_2 = ['release_speed', 'release_pos_x',
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

#updating dataframe
df3 = df2.drop(missing_list_2, axis=1)


# %%
#transforming a few remiaing categorical variables to dummies
df3['batter_is_right'] = np.where(df2['stand'] == 'R', 1, 0)
df3['pitcher_is_right'] = np.where(df2['p_throws'] == 'R', 1, 0)
df3['inning_top'] = np.where(df2['inning_topbot'] == 'Top', 1, 0)
df3 = df3.drop(['stand', 'p_throws', 'inning_topbot'], axis=1)


# %%


#transforming fielding alignment
transformed_data = pd.get_dummies(df3, columns=['if_fielding_alignment', 'of_fielding_alignment'], drop_first=True)

path = 
pd.to_csv()


