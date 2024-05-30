import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
from preprocess_common import preprocess_df
from prompt_toolkit import print_formatted_text, HTML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB


DATASET_DEFAULT_LOCATION = "../dataset/songsTaylorSwiftAndRihanna.xlsx"


def main():
    parser = argparse.ArgumentParser(description="Clickbait NB")
    parser.add_argument(
        "--remove-stopwords", action="store_true", help="Remove stopwords from dataset"
    )
    args = parser.parse_args()

    # Read data
    df = pd.read_excel(DATASET_DEFAULT_LOCATION)
    df = df.sample(frac=1).reset_index(drop=True)

    # Preprocess data
    preprocess_df(df, args.remove_stopwords)
    features = df.title.to_list()
    labels = df.label.to_list()

    # Split train, validation, test. train = 90%, test = 10%
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.9, random_state=42
    )

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # Build the model
    model = GaussianNB()

    # Train the model using the training data
    model.fit(x_train.toarray(), y_train)

    # Predict the categories of the test data
    predictions = model.predict(x_test.toarray())

    print_formatted_text(HTML("<b><u>Confusion matrix</u></b>"))
    matrix = torch.zeros([2, 2], dtype=torch.int32)
    for pred, lab in zip(predictions, y_test):
        matrix[pred, lab] += 1

    print(matrix)

    print_formatted_text(HTML("<b><u>Classification report</u></b>"))
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
