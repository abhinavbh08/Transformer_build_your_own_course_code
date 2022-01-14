# Remove the duplicates and split the data into train, test and val sets.
from os import write
from sklearn.model_selection import train_test_split
import numpy as np

def write_file(file_name, data):
    with open('data/de-en_deduplicated/'+file_name, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

with open("data/de-en_deduplicated/WMT-News.de-en.en", "r") as f:
    en = f.read().split("\n")[:-1]

with open("data/de-en_deduplicated/WMT-News.de-en.de", "r") as f:
    de = f.read().split("\n")[:-1]

# en_distinct = []
# de_distinct = []
# for index, line in enumerate(en):
#     if line not in en_distinct:
#         en_distinct.append(line)
#         de_distinct.append(de[index])

write_file("de-en.en", en)
write_file("de-en.de", de)

# Open the english dataset
with open("data/de-en_deduplicated/de-en.en", "r") as file:
    data_en = file.read().split("\n")[:-1]

# Open the German dataset
with open("data/de-en_deduplicated/de-en.de", "r") as file:
    data_de = file.read().split("\n")[:-1]

# Do train-test split from the full dataset.
train_en, test_en, train_de, test_de = train_test_split(data_en, data_de, test_size=1000, random_state=42)

# So train-val split from the training dataset.
train_en, val_en, train_de, val_de = train_test_split(train_en, train_de, test_size=1000, random_state=42)

write_file("train.en", train_en)
write_file("train.de", train_de)
write_file("val.en", val_en)
write_file("val.de", val_de)
write_file("test.en", test_en)
write_file("test.de", test_de)