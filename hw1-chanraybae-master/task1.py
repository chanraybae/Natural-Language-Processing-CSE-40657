import unigram
import utils

# data from a file and return a list of lists of characters
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [list(line.rstrip('\n')) for line in f]


# Using the unigram model
def predict_most_probable_char(model, prefix):
    _, logprobs = model.step(None, model.vocab.numberize(prefix[-1] if prefix else '<BOS>'))
    return model.vocab.denumberize(logprobs.index(max(logprobs)))


def main():
    # Read training data and train a unigram model
    training_data = load_data("data/large")
    model = unigram.Unigram(training_data)

    # For each character position in dev data, findmost probable character
    dev_data = load_data("data/dev")
    corr_pred = 0
    tot_pred = 0

    for line in dev_data:
        for idx, char in enumerate(line):
            prefix = line[:idx]
            predicted_char = predict_most_probable_char(model, prefix)
            if predicted_char == char:
                corr_pred += 1
            tot_pred += 1

    # 3. Report the number of correct characters, the total number of characters, and the accuracy
    accuracy = (corr_pred / tot_pred) * 100
    print(f"Correct Predictions: {corr_pred}")
    print(f"Total Predictions: {tot_pred}")
    print(f"Accuracy: {accuracy:.3f}%")


if __name__ == "__main__":
    main()
