import pandas as pd
import numpy as np
import nltk

def preprocess_df(df, remove_stopwords):
    df['title'] = df['title'].astype(str)
    df['title'].fillna('', inplace=True)

    if remove_stopwords:
        from nltk.corpus import stopwords
        stop = set(stopwords.words('english'))
        df['title'] = df['title'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stop]))
    return df


def pad_data(data, seq_length):
    padded_data = np.zeros((len(data), seq_length), dtype=int)

    for i, row in enumerate(data):
        padded_data[i, -len(row) :] = np.array(row)[:seq_length]

    return padded_data


def data_to_padded_sequences(titles, word_to_id, seq_len):
    # Convert titles to sequences
    data_sequences = []
    for title in titles:
        data_sequences.append([word_to_id.get(word, 0) for word in title.split()])

    # Pad sequences with zeros
    padded_data = pad_data(data_sequences, seq_len)

    return padded_data
