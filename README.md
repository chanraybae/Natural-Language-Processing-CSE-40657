# Natural-Language-Processing-CSE-40657
Current course, will upload more as I get graded feedback

C + V HW1 instructions if HTML file download is undesired:


CSE 40657/60657
Homework 1
Due
2023/09/15
Points 30
In situations where text input is slow (mobile phones, Chinese/Japanese characters, users with disabilities), it can be helpful for the computer to be able to guess the next character(s) the user will type. In this assignment, you'll build a character language model and test how well it can predict the next character.

Setup
Visit this GitHub Classroom link to create a Git repository for you, and clone it to your computer. It contains the following files:

data/small	small training data
data/large	full training data
data/dev	development data
data/test	test data
predict.py	text-prediction demo
unigram.py	unigram language model
utils.py	possibly useful code
The data files come from the NUS SMS Corpus, a collection of real text messages sent mostly by students at the National University of Singapore. This is the English portion of the corpus, though it has a lot of interesting examples of Singlish, which mixes in elements of Malay and various Chinese languages.

In the following, point values are written after each requirement, like this.30

All the language models that you build for this assignment should support the following operations. If the model object is called m, then m should have a member m.vocab of type utils.Vocab, which converts characters to integers (m.vocab.numberize) and converts integers to characters (m.vocab.denumberize). Furthermore, it should implement the following methods:

m.start(): Return the start state.
m.step(q, a): Run one step of the model, where q is the state the model is in before the step, and a is the numberized input symbol. The return value is (r, p), where r is the state the model enters after the step, and p is the model's prediction of the next output symbol, such that for any numberized symbol b, p[b] is the log-probability of b.
For example, the code to compute the log-probability of foo would be
q = m.start()
p_foo = 0.
q, p = m.step(q, m.vocab.numberize('<BOS>'))
p_foo += p[m.vocab.numberize('f')]
q, p = m.step(q, m.vocab.numberize('f'))
p_foo += p[m.vocab.numberize('o')]
q, p = m.step(q, m.vocab.numberize('o'))
p_foo += p[m.vocab.numberize('o')]
q, p = m.step(q, m.vocab.numberize('o'))
p_foo += p[m.vocab.numberize('<EOS>')]
A model implementing this interface can be plugged into predict.py, which predicts the next 20 characters based on what you've typed so far.
1. Baseline
The file unigram.py provides a class Unigram that implements the above interface using a unigram language model. The Unigram constructor expects a list of lists of characters to train on.

Write a program that reads in the training data (data/large) and uses unigram.Unigram to train a unigram model.3 Be sure to strip off trailing newlines when you read a file.
Write code that, for each character position in the development data (data/dev), including EOS, predicts the most probable character given all previous correct characters.5 Report the number of correct characters, the total number of characters, and the accuracy (what percent of the characters are correct).1 It should be 6620/40176 ≈ 16.477%.1
Try running python predict.py data/large By default, it uses a unigram language model, which is not very interesting, because the model always predicts a space. (Nothing to report here.)
2. n-gram language model
In this part, you'll replace the unigram language model with a 5-gram model.

Implement a 5-gram language model.5 For smoothing, just use add-one smoothing.3
Train it on data/large. Report your accuracy on data/dev, which should be at least 49%.1 Remember to include EOS but not BOS when computing accuracy.
After you've gotten your model working, run it on the test set (data/test) and report your accuracy, which should be at least 49%.1
Try running predict.py with your model. (Nothing to report.)
3. RNN language model
Now we will try building a neural language model using PyTorch.

To get started with PyTorch, try our tutorial notebook, which trains a unigram language model. For help with Kaggle, please see our Kaggle tutorial.
Write code to implement an RNN language model.5 You may reuse any code from the tutorial notebook, and you may use any functions provided by PyTorch. It turns out that a simple RNN isn't very sensitive to dependencies between the previous state and the input symbol. An LSTM (long short term memory) RNN works better (in place of equation 2.17). It computes two sequences of vectors,
(h(−1),c(−1))(h(t),c(t))=(0,0)=LSTMCell(x(t),(h(t−1),c(t−1)))
where LSTMCell
 is an instance of PyTorch's LSTMCell. The x(t)
 can be one-hot vectors as in equation 2.17, or they can be word embeddings. The outputs are just the h(t)
, which plug into equation 2.21.
Train on data/small and validate on data/dev, using d=128
 for both h(t)
 and c(t)
. Report your dev accuracy, which should be at least 40%.1 For us, using no GPU, each epoch took less than 5 minutes, and 10 epochs was enough.
Train on data/large and again validate on data/dev, using d=512
. Report your dev accuracy, which should be at least 53%.1 For us, using a P100 GPU, each epoch took about 20 minutes, and 10 epochs was enough. Save and submit your best model.1
After you've gotten your model working, run it on the test set (data/test) and report your accuracy,1 which should be at least 53%.1
Try running predict.py with your model. Because training takes a while, you'll probably want to load a trained model from disk. (Nothing to report.)
Submission
Please read these submission instructions carefully.

As often as you want, add and commit your submission files to the repository you created in the beginning.
Before submitting, the repository should contain:
All of the code that you wrote.
Your final saved model from Part 3.
A README.md file with
Instructions on how to build/run your code.
Your responses to all of the instructions/questions in the assignment.
To submit:
Push your work to GitHub and create a release in GitHub by clicking on "Releases" on the right-hand side, then "Create a new release" or "Draft a new release". Fill in "Tag version" and "Release title" with the part number(s) you're submitting and click "Publish Release".
If you submit the same part more than once, the grader will grade the latest release for that part.
For computing the late penalty, the submission time will be considered the commit time, not the release time.
© 2015–2021 David Chiang. Unless otherwise indicated, all materials are licensed under a Creative Commons Attribution 4.0 International License.
