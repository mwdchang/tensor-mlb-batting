import parser as parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import sys

import collections
Dataset = collections.namedtuple('Dataset', ['data', 'target'])

from flask import Flask, request, jsonify


def dict2List(batter):
  return [
    batter["H"],
    batter["2B"],
    batter["3B"],
    batter["HR"],
    batter["RBI"],
    batter["R"]
  ]

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
  
# Shuffle data
np.random.shuffle(batters_array)


# Divide into features and target
for batter in batters_array:
  raw_feature.append(dict2List(batter))
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


# Model
feature_columns = [tf.feature_column.numeric_column("x", shape=[6])]
classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[10, 30, 10],
  n_classes=4,
  model_dir="save")


# Train
print("Training...")
train_input_fn = create_input_fn(train_data, train=True)
classifier.train(input_fn=train_input_fn, steps=2000)


# Accuracy
print("Checking accuracy")
test_input_fn = create_input_fn(test_data, train=False)
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))



# Flask
app = Flask(__name__, static_url_path='', static_folder='')

@app.route('/sample')
def sample():
  sample = np.random.choice(batters_array, 4)
  sample_array = []
  for batter in sample:
    sample_array.append(dict2List(batter))
  new_samples = np.array(sample_array, dtype=np.int)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn( 
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)
  predictions = list(classifier.predict(input_fn=predict_input_fn))
  probs = [p["probabilities"].tolist() for p in predictions]

  c = 0
  for batter in sample:
    batter["_pred"] = probs[c]
    c = c + 1
  return jsonify(list(sample))


  
@app.route('/')
def root():
  return app.send_static_file('index.html')


if __name__ == '__main__':  # pragma: no cover
  app.run(port=5000)


