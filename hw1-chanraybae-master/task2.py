class NgramModel:
    def __init__(self, n=5):
        self.n = n
        self.vocab = set()  # for unique characters
        self.ngram_counts = {}  # for n-gram counts
        self.context_counts = {}  # for (n-1)-gram counts

    def train(self, data):
        for line in data:
            # accounting boundary symbols to line
            line = ['<BOS>'] * (self.n - 1) + line + ['<EOS>']

            # populating vocab
            for char in line:
                self.vocab.add(char)

            # counting n-grams and (n-1)-grams
            for i in range(len(line) - self.n + 1):
                ngram = tuple(line[i:i + self.n])
                context = ngram[:-1]

                # updating ngram counts
                if ngram in self.ngram_counts:
                    self.ngram_counts[ngram] += 1
                else:
                    self.ngram_counts[ngram] = 1

                # updating (n-1)-gram counts
                if context in self.context_counts:
                    self.context_counts[context] += 1
                else:
                    self.context_counts[context] = 1

    def pred_next_char(self, context):
        # confirming context is correct size
        assert len(context) == self.n - 1

        max_prob = 0.0
        best_char = None

        for char in self.vocab:
            ngram = tuple(context + [char])

            # add-one smoothing
            ngram_count = self.ngram_counts.get(ngram, 0) + 1
            context_count = self.context_counts.get(tuple(context), 0) + len(self.vocab)

            prob = ngram_count / context_count

            if prob > max_prob:
                max_prob = prob
                best_char = char

        return best_char

    def get_acc(self, data):
        corr_pred = 0
        tot_pred = 0

        for line in data:
            for i in range(len(line) - self.n + 1):
                context = line[i:i + self.n - 1]
                actual_next_char = line[i + self.n - 1]
                pred_next_char = self.pred_next_char(context)

                tot_pred += 1
                if pred_next_char == actual_next_char:
                    corr_pred += 1

        return corr_pred / tot_pred


if __name__ == "__main__":
    # loading training data
    train_data = [list(line.rstrip('\n')) for line in open("data/large", encoding='utf-8')]

    model = NgramModel(5)
    model.train(train_data)

    # eval on dev data
    dev_data = [list(line.rstrip('\n')) for line in open("data/dev", encoding='utf-8')]
    dev_acc = model.get_acc(dev_data)
    print(f"Accuracy on dev data: {dev_acc * 100:.2f}%")

    # eval on test data albeit only after confirming acc on dev data
    test_data = [list(line.rstrip('\n')) for line in open("data/test", encoding='utf-8')]
    test_acc = model.get_acc(test_data)
    print(f"Accuracy on test data: {test_acc * 100:.2f}%")
