import parser as parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import sys

import collections
Dataset = collections.namedtuple('Dataset', ['data', 'target'])



rankings = parser.processHoF()
batters  = parser.processBatting()
people   = parser.processPeople()


raw_feature = []
raw_target = []

# Add ranking and remove batters with too few at bats
batters_array = []
for key in batters:
  batter = batters[key]
  if key not in rankings:
    batter["RANK"] = 0
  else:
    batter["RANK"] = rankings[key]

  if batter["AB"] < 1000:
    continue 
  batters_array.append(batter)
  
np.random.shuffle(batters_array)

# Remove batters with too few at-bats
for batter in batters_array:
  raw_feature.append([
    batter["H"],
    batter["2B"],
    batter["3B"],
    batter["HR"],
    batter["RBI"],
    batter["R"]
  ])
  raw_target.append([
    batter["RANK"]
  ])


# Split 
cut = 3200
train_x = raw_feature[:cut]
train_y = raw_target[:cut]
test_x = raw_feature[cut:]
test_y = raw_target[cut:]

train_data = Dataset(data=train_x, target=train_y)
test_data = Dataset(data=test_x, target=test_y)

print('data', len(train_x), len(train_y), len(test_x), len(test_y))


def create_input_fn(data, train=True):
    epoch = None
    shuffle = True

    if (train == False):
        epoch = 1
        shuffle = False

    input = tf.estimator.inputs.numpy_input_fn(
      x = {"x": np.array(data.data)},
      y = np.array(data.target),
      num_epochs = epoch, shuffle = shuffle)
    return input



# train = tf.contrib.learn.datasets.base.load_csv_without_header(
#   filename="mlb-data.csv",
#   target_dtype=np.int,
#   features_dtype=np.int)


feature_columns = [tf.feature_column.numeric_column("x", shape=[6])]

classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[10, 30, 10],
  n_classes=4,
  model_dir="save")

#train_input_fn = create_input_fn(train, train=True)
train_input_fn = create_input_fn(train_data, train=True)
classifier.train(input_fn=train_input_fn, steps=2000)


test_input_fn = create_input_fn(test_data, train=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


# new_samples = np.array(
#       [
#        [9800, 70, 15, 50, 800, 1200],
#        [1300, 30, 10, 10, 140, 500],
#       ], dtype=np.int)
# 
# predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={"x": new_samples},
#       num_epochs=1,
#       shuffle=False)
# 
# predictions = list(classifier.predict(input_fn=predict_input_fn))
# print("predictions", predictions)
# 
# predicted_classes = [p["classes"] for p in predictions]
# print("New Samples, Class Predictions: {}\n".format(predicted_classes))


