import os
import csv


def get_training_data(dir_path, prop=1):
    with open(os.path.join(dir_path, 'train.csv'), 'r', encoding='utf-8') as csvfile:
        training_data = [row for row in csv.DictReader(csvfile, delimiter=',')]
        training_data = training_data[0:int(prop * len(training_data))]
        for entry in training_data:
            with open(os.path.join(dir_path, 'train', entry['FileIndex'] + '.txt'), 'r', encoding='utf-8') as reviewfile:
                entry['Review'] = reviewfile.read()

    indices = [row['FileIndex'] for row in training_data]
    reviews = [row['Review'] for row in training_data]
    labels = [int(row['Category']) for row in training_data]
    return indices, reviews, labels