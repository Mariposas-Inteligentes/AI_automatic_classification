import torch


class Clickbait_RNN(torch.nn.Module):
    def __init__(self, embedding_matrix, hidden_size, n_layers):
        super(Clickbait_RNN, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding_matrix)
        self.lstm = torch.nn.LSTM(
            embedding_matrix.shape[1],
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x, hidden):
        embeddings = self.embedding(x)
        lstm_output, (hn, cn) = self.lstm(embeddings, hidden)
        out = self.linear(hn[-1])
        y = self.sig(out)

        return y
