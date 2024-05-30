import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import nltk
from sklearn.metrics import classification_report
import argparse
import gensim
from clickbait_rnn import Clickbait_RNN
from preprocess_common import preprocess_df, data_to_padded_sequences
from prompt_toolkit import print_formatted_text, HTML


EMBEDDING_DIM = 300
DATASET_DEFAULT_LOCATION = "../dataset/songsTaylorSwiftAndRihanna.xlsx"
SUPPORT_PATH = "./support/"


def word_to_id_map_get(vocab):
    word_to_id = {}
    word_to_id_file = Path(SUPPORT_PATH + "word_to_id.json")
    if word_to_id_file.is_file():
        print("Using an already existing word to id index file")
        word_to_id = json.load(open(word_to_id_file, "r"))
    else:
        print("Creating word to id index...", end="", flush=True)
        word_to_id = {word: i for i, word in enumerate(vocab, 1)}

        Path(SUPPORT_PATH).mkdir(parents=True, exist_ok=True)
        json.dump(word_to_id, open(word_to_id_file, "w"), indent=4)
        print("[DONE]")

    return word_to_id


def embedding_matrix_get(word_to_id, word_to_embedding):
    # Create embedding matrix
    print("Creating embedding matrix...", end="", flush=True)

    vocab_size = len(word_to_id) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    for word, i in word_to_id.items():
        if word in word_to_embedding.wv:
            embedding_matrix[i] = word_to_embedding.wv[word]

    print("[DONE]")

    return embedding_matrix


def get_data_from_df(df, remove_stopwords, seq_len):
    preprocess_df(df, remove_stopwords)
    
    print("Training word embeddings...", end="", flush=True)
    sentences = [title.split() for title in df["title"]]
    word_to_embedding = gensim.models.Word2Vec(sentences, size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
    print("[DONE]")

    # Build word to id mapping
    vocab = set(" ".join([i for i in df["title"]]).split())
    word_to_id = word_to_id_map_get(vocab)

    # Build embedding matrix
    embedding_matrix = embedding_matrix_get(word_to_id, word_to_embedding)

    # Convert data (titles) to padded sequences
    data = data_to_padded_sequences(df.title.values, word_to_id, seq_len)

    return data, embedding_matrix


def get_labels_from_df(df):
    return df.label.to_list()


def train(epoch, model, hidden, optimizer, loss_fn, train_loader):
    losses = []
    for i, (x, y) in enumerate(train_loader):
        # Call our model
        y_predicted = model(x, hidden)
        loss = loss_fn(y_predicted, y.view(-1, 1))

        # Propagate loss, optimization step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def valid(epoch, model, hidden, loss_fn, valid_loader):
    losses = []
    for i, (x, y) in enumerate(valid_loader):
        y_predicted = model(x, hidden)
        loss = loss_fn(y_predicted, y.view(-1, 1))

        losses.append(loss.item())

    return np.mean(losses)


def test(model, hidden, test_loader):
    predictions = []
    labels = []
    for i, (x, y) in enumerate(test_loader):
        y_predicted = model(x, hidden)
        predictions.extend(torch.round(y_predicted.flatten()).int().tolist())
        labels.extend(y.int().tolist())

    print_formatted_text(HTML("<b><u>Confusion matrix</u></b>"))
    matrix = torch.zeros([2, 2], dtype=torch.int32)
    for pred, lab in zip(predictions, labels):
        matrix[pred, lab] += 1

    print(matrix)

    print_formatted_text(HTML("<b><u>Classification report</u></b>"))
    print(classification_report(labels, predictions))


def main():
    parser = argparse.ArgumentParser(description="Clickbait RNN")
    parser.add_argument(
        "--epochs", type=int, default=10, nargs="?", help="Number of epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=256, help="Hidden state size for LSTM layer"
    )
    parser.add_argument(
        "--remove-stopwords", action="store_true", help="Remove stopwords from dataset"
    )
    parser.add_argument(
        "--sequence-len", type=int, default=34, help="Lenght of the input sequence"
    )
    parser.add_argument(
        "--lstm-layers", type=int, default=1, help="Number of layers for the LSTM layer"
    )
    parser.add_argument(
        "--test", action="store_true", help="Use test set to get performance metrics"
    )
    parser.add_argument("--save-model", type=str, help="Save model to specified path")

    args = parser.parse_args()

    # Read data
    df = pd.read_excel(DATASET_DEFAULT_LOCATION)
    data, embedding_matrix = get_data_from_df(
        df, args.remove_stopwords, args.sequence_len
    )
    labels = get_labels_from_df(df)

    # Split train, validation, test. train = 80%, validation = 10%, test = 10%
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        data, labels, train_size=0.8, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_remaining, y_remaining, test_size=0.5, random_state=42
    )

    # Datasets
    train_data = TensorDataset(torch.tensor(x_train), torch.Tensor(y_train))
    valid_data = TensorDataset(torch.tensor(x_val), torch.Tensor(y_val))
    test_data = TensorDataset(torch.tensor(x_test), torch.Tensor(y_test))

    # Dataloaders
    batch_size = args.batch_size
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, drop_last=True
    )
    valid_loader = DataLoader(
        valid_data, shuffle=True, batch_size=batch_size, drop_last=True
    )
    test_loader = DataLoader(
        test_data, shuffle=True, batch_size=batch_size, drop_last=True
    )

    # Create model
    n_lstm_layers = args.lstm_layers
    hidden_size = args.hidden_size
    model = Clickbait_RNN(
        torch.FloatTensor(embedding_matrix), hidden_size, n_lstm_layers
    )

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        # Train
        h0 = torch.zeros(n_lstm_layers, batch_size, hidden_size)
        c0 = torch.zeros(n_lstm_layers, batch_size, hidden_size)
        train_loss = train(epoch, model, (h0, c0), optimizer, loss_fn, train_loader)

        # Validation
        h0 = torch.zeros(n_lstm_layers, batch_size, hidden_size)
        c0 = torch.zeros(n_lstm_layers, batch_size, hidden_size)
        valid_loss = valid(epoch, model, (h0, c0), loss_fn, valid_loader)

        print(
            "Epoch: {}, Train loss: {:0.4f}, Validation loss: {:0.4f}".format(
                epoch, train_loss, valid_loss
            )
        )

    # Test
    if args.test:
        h0 = torch.zeros(n_lstm_layers, batch_size, hidden_size)
        c0 = torch.zeros(n_lstm_layers, batch_size, hidden_size)
        test(model, (h0, c0), test_loader)

    # Save model
    if args.save_model:
        with open(args.save_model, "wb") as f:
            torch.save(model, f)


if __name__ == "__main__":
    main()
