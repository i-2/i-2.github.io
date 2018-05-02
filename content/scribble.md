Title: Simple Scribble Detection by k-nearest neighbours using python
Date: 2018-05-02 11:20
Category: Python
Tags: machine-learning, python
Slug: scribble-detection
Authors: Sathya Narrayanan
Summary: K-nearest neighbours(K-NN) is one of the simplest machine learning algorithms. Using K-NN we are going to classify whether the given text is a scribble or not.

I am a beginner to the machine learning space. I felt it very hard to understand the mathematical essence assosiated with each learning algorithms. One day a friend of mine introduced me to this (k-NN) approach. Unlike other learning algorithms this is very simple and easy to understand. So let's get started!

## What is K-Nearest Neighbours ?

"In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space."

I know you are mad at me now. Let's say it in a simple terms.

If there are people standing in 5 queues, most people would prefer to stand in queue where there are few people before them, or in other words they prefered a queue which is near to the counter. It is this notion of distance the K-NN algorithm will work.

If you need to write your own k-nn algorithm. You should first come up with a definition of distance for your data.

## Let's dive in to the example

We are going to classify text whether it is a scribble or not. 
Entire code is available in my [github repo](https://github.com/sourcepirate/scribble-detection).
We are going to use a dataset of words.
below is a sample set.

```md

# scribble_data.txt
sno,WORDS,scribble,label
0,human,False,word
1,animal,False,word
2,frog,False,word
.
.
.
197,dsncfiwer,True,scribble
198,ckdnsosdfs,True,scribble
199,ewiojocwmo,True,scribble

```

We are going to use this dataset for the demo.

## Notion of distance

If you take a closer look in your keyboard layout. Three row are available

```md
qwertyuiop
asdfghjkl
zxcvbnm
```
so we can use this information to create a vector. For example,

```md

u = upper, m = middle , l= lower

hello = 2u + 3m + 0l = [2, 3, 0]
cat = 1u  + 1m + 1l = [1, 1, 1]

```

Now we have our vectors available. we calculate the distance between them

```python

a = [1, 2, 3]
b = [1, 3, 1]

magnitude = (a[0] - b[0]) ^ 2 + (a[1] - b[1]) ^ 2 + (a[2] - b[2]) ^ 2
distance = sqrt(magnitude)

```

Now we have our distance. now lets sum up all these to python code.

```python

import csv
from collections import Counter
from math import sqrt

UPPER = "qwertpoiuy"
MIDDLE = "asdfglkjh"
LOWER = "zxcvbnm"

def get_class_count(word_dict, _kls):
    """getting a class count"""
    keys = set(word_dict.keys())
    _kls_set = set(_kls)
    intersection = keys & _kls_set
    return sum([word_dict[key] for key in intersection])


def get_stats(word):
    """ Representign a word as a vertor [i, j, k]
        i - Number of words from upper keyboard row
        j - Number of words from middle keyboard row
        k - Number of words from lower keyboard row
    """
    stat = Counter(word)
    return [get_class_count(stat, UPPER),
            get_class_count(stat, MIDDLE),
            get_class_count(stat, LOWER)]

def distance(word1, word2):
    """ Does the vector magnitude for two stats
    """
    stat1 = get_stats(word1)
    stat2 = get_stats(word2)
    return sqrt(sum([(i-j)**2 for i, j in zip(stat1, stat2)]))

```

## Determining the closest

Let's just recall what columns do we have in our dataset.

```md

WORDS
label
scribble

```

This is called training dataset. The main purpose or training dataset is to train
you machine on data.

What does this algoirthm do when it gets trained ?
Answer: If we give it a word to classify it checks whether which point is near to it. In our case either 
it can be a word point or scibble point. If a word point it near to it, then the given word is a word,
else it is a scribble.

Main problem with this K-NN approach is you need a lot of data to train your model to be perfect.
out dataset has only 200 rows. so don't expect it to work well on production.

code for selecting the nearest node.

```python

def majority(labels):
    """get the majority for label
       Since we have to filter only one optimally
       if two labels have same counts try
       removing the farthest one and
       continue it.
    """
    counts = Counter(labels)
    winner, winner_count = counts.most_common(1)[0]
    num_winners = len([count for count in counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority(labels[:-1])

def knn_classify(k, labeled, word):
    """each labeled point should be a pair (point, label)"""
    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled,
                         key=lambda point: distance(point["WORDS"], word))

    # find the labels for the k closest
    k_nearest_labels = [label["label"] for label in by_distance[:k]]

    # and let them vote
    return majority(k_nearest_labels)

```

majority function takes in the list of labels and gives out marjority in give labels. If you two or more labels has the same majority we remove the farthest for the list of labels and run majority again(There can be only one winner).

knn_classify function takes in :

**k** : which denotes the no of nearest nodes we need to see. In our case we choose to go with 2.

**labeled**: the training data set.

**word**: the word to be classified.


## Now the fun part. Demo Time

```python
reader = csv.DictReader(open("scribble_data.txt"))
values = list(reader)
test_scribble = ["sdjkflsd", 
                "ivfnfin",
                "siofuewoi",
                "ewuroqo",
                "weiuoro",
                "vbfbjsad",
                "hello",
                "word",
                "wouerwiuer",
                "jupiter",
                "simplex"]
for data in test_scribble:
    predicted = knn_classify(2, values, data)
    print("{} is classified as {}".format(data, predicted))
```

Now lets see the output

```
sdjkflsd is classified as scribble
ivfnfin is classified as scribble
siofuewoi is classified as scribble
ewuroqo is classified as scribble
weiuoro is classified as scribble
vbfbjsad is classified as scribble
hello is classified as word
word is classified as word
wouerwiuer is classified as scribble
jupiter is classified as word
simplex is classified as word

```

It pretty much works good. So we are done.
