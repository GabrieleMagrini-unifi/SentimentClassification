import pandas as pd
import re

data = pd.DataFrame(columns=('sentiment', 'review'))


for i in range(1, 1001):
    review = open("data/neg/neg ("+str(i)+").txt")
    string_without_line_breaks = ""
    for line in review:
        stripped_line = line.rstrip()
        string_without_line_breaks = string_without_line_breaks + stripped_line
    review = string_without_line_breaks
    data.loc[i] = [0,  review]
for i in range(1, 1001):
    review = open("data/pos/pos ("+str(i)+").txt")
    string_without_line_breaks = ""
    for line in review:
        stripped_line = line.rstrip()
        string_without_line_breaks = string_without_line_breaks + stripped_line
    review = string_without_line_breaks
    data.loc[i+1000] = [1, review]

data = data.sample(frac=1).reset_index(drop=True)

data.to_csv('data_review.tsv', sep='\t')

