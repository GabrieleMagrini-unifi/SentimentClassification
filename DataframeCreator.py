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
    data.loc[(i-1)*2] = [0,  review]
    review2 = open("data/pos/pos (" + str(i) + ").txt")
    string_without_line_breaks = ""
    for line in review2:
        stripped_line = line.rstrip()
        string_without_line_breaks = string_without_line_breaks + stripped_line
    review2 = string_without_line_breaks
    data.loc[(i-1)*2+1] = [1, review2]

#data = data.sample(frac=1).reset_index(drop=True)

data.to_csv('data/data_review_balanced_reduced.tsv', sep='\t')

print(data)
