# %%
from pull_data import pull_pitcher_data

first = 'blake'
last = 'snell'
start = '2023-04-30'
end = '2023-10-01'


df = pull_pitcher_data(first, last, start, end)
file_name = first + "_" + last + ".csv"
path = '../data/raw/' + file_name



df.to_csv(path)


