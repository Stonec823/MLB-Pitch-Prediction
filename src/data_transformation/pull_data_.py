# %%
from preprocess import pull_pitcher_data

import os

first = 'max'
last = 'fried'
start = '2024-04-05'
end = '2024-10-01'


df = pull_pitcher_data(first, last, start, end)
# file_name = first + "_" + last + ".csv"
# path = '/data/raw/' + file_name


file_name = first + "_" + last + ".csv"


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
raw_data_dir = os.path.join(project_root, "data", "raw")
path = os.path.join(raw_data_dir, file_name)


os.makedirs(raw_data_dir, exist_ok=True)


df.to_csv(path, index=False)


# df.to_csv(path)


