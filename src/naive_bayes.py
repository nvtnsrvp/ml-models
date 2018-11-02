import collections
import matplotlib.pyplot as plt
import numpy as np
import util


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    return map(lambda w: w.lower(), message.split(" "))


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    msgs = [get_words(msg) for msg in messages]
    counter = collections.Counter(w for msg in set(msgs) for w in msg)
    words = [w for w, c in counter.items() if c >= 5]
    return dict(zip(words, range(len(words))))


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    x = np.zeros([len(messages), len(word_dictionary)])
    msgs = [get_words(msg) for msg in messages]
    for i in range(len(msgs)):
        for w in msgs[i]:
            if w in word_dictionary:
                x[i, word_dictionary[w]] += 1
    return x


def main(train_path, test_path):
    train_messages, train_labels = util.load_spam_dataset(train_path)
    test_messages, test_labels = util.load_spam_dataset(test_path)

    dictionary = create_dictionary(train_messages)
    util.write_json('./output/p06_dictionary', dictionary)

    x_train = transform_text(train_messages, dictionary)
    x_test = transform_text(test_messages, dictionary)
    np.savetxt('./output/p06_sample_train_matrix', x_train[:100,:])

    nb = NaiveBayes()
    nb.fit(x_train, train_labels)
    y_pred = nb.predict(x_test)
    np.savetxt('./output/p06_naive_bayes_predictions', y_pred)

    accuracy = np.mean(y_pred == test_labels)
    print('Naive Bayes had an accuracy of {} on the testing set'.format(accuracy))

    top_5_words = nb.top_words(5, dictionary)
    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)


class NaiveBayes():
    def fit(self, x, y):
        """Fit a naive bayes model.

        This function should fit a Naive Bayes model given a training matrix and labels.

        Args:
            x: A numpy array containing word counts for the training data
            y: The binary (0 or 1) labels for that training data

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        y1 = np.sum(y)
        p_y1 = 1.0/m * y1
        self.p_y1 = np.log(p_y1)
        self.p_y0 = np.log(1-p_y1)

        self.p_x_y1 = np.log(1.0/(y1+n) * (np.sum(x[y == 1], axis=0)+1))
        self.p_x_y0 = np.log(1.0/(m-y1+n) * (np.sum(x[y == 0], axis=0)+1))

    def predict(self, x):
        """Use a Naive Bayes model to compute predictions for a target matrix.

        This function should be able to predict on the models that fit_naive_bayes_model
        outputs.

        Args:
            x: A numpy array containing word counts

        Returns: A numpy array containing the predictions from the model
        """
        p_y1_x = x.dot(self.p_x_y1) + self.p_y1
        p_y0_x = x.dot(self.p_x_y0) + self.p_y0
        return (p_y1_x > p_y0_x).astype(int)

    def top_words(self, k, dictionary):
        """Compute the top k words that are most indicative of the spam (i.e positive) class.

        Use the metric given in 6c as a measure of how indicative a word is.
        Return the words in sorted form, with the most indicative word first.

        Args:
            k: An integer indicating the number of words to display
            dictionary: A mapping of word to integer ids

        Returns: The top k most indicative words in sorted order with the most indicative first
        """
        to_word = dict((i, w) for w, i in dictionary.items())
        norm_p_x_y1 = list(zip(self.p_x_y1-self.p_x_y0, range(self.p_x_y1.size)))
        norm_p_x_y1.sort()
        return [to_word[norm_p_x_y1[i][1]] for i in range(max(-len(norm_p_x_y1), -k), 0)]


if __name__ == "__main__":
    main(train_path='../data/ds6_train.tsv',
         test_path='../data/ds6_test.tsv')
