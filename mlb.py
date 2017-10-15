import parser as parser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
import numpy as np


rankings = parser.processHoF()
batters  = parser.processBatting()
people   = parser.processPeople()


with open("mlb-data.csv", "w") as f:
  for key in batters:
    if key not in rankings:
      batters[key]["RANK"] = 0
    else:
      batters[key]["RANK"] = rankings[key]

    if batters[key]["AB"] < 1000:
      continue

    f.write(
      str(batters[key]["H"]) + "," + 
      str(batters[key]["2B"]) + "," +
      str(batters[key]["3B"]) + "," + 
      str(batters[key]["HR"]) + "," + 
      str(batters[key]["RBI"]) + "," + 
      str(batters[key]["R"]) + "," + 
      str(batters[key]["RANK"]) + "\n")



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




train = tf.contrib.learn.datasets.base.load_csv_without_header(
  filename="mlb-data.csv",
  target_dtype=np.int,
  features_dtype=np.int)

feature_columns = [tf.feature_column.numeric_column("x", shape=[6])]

classifier = tf.estimator.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[10, 30, 10],
  n_classes=4,
  model_dir="save")


train_input_fn = create_input_fn(train, train=True)
classifier.train(input_fn=train_input_fn, steps=2000)

new_samples = np.array(
      [
       [3800, 700, 150, 480, 1900, 2100],
       [1300, 30, 10, 10, 140, 500],
      ], dtype=np.int)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]
print("New Samples, Class Predictions: {}\n".format(predicted_classes))



"""
playerID
yearid
votedBy
ballots
needed
votes
inducted
category
needed_note
"""
