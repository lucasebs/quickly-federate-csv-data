import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

csv_url = "https://docs.google.com/spreadsheets/d/1eJo2yOTVLPjcIbwe8qSQlFNpyMhYj-xVnNVUTAhwfNU/gviz/tq?tqx=out:csv"

df = pd.read_csv(csv_url, na_values=("?",))

client_id_colname = 'native.country' # the column that represents client ID
SHUFFLE_BUFFER = 1000
NUM_EPOCHS = 1

# split client id into train and test clients
client_ids = df[client_id_colname].unique()
train_client_ids = pd.DataFrame(client_ids).sample(frac=0.5).values.tolist()
print(train_client_ids)
test_client_ids = [x for x in client_ids if x not in train_client_ids]
print(test_client_ids)

def create_tf_dataset_for_client_fn(client_id):
  # a function which takes a client_id and returns a
  # tf.data.Dataset for that client
  client_data = df[df[client_id_colname] == client_id[0]]
  dataset = tf.data.Dataset.from_tensor_slices(client_data.fillna('').to_dict("list"))
  dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(1).repeat(NUM_EPOCHS)
  return dataset

train_data = tff.simulation.datasets.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
test_data = tff.simulation.datasets.ClientData.from_clients_and_fn(
        client_ids=test_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )

example_dataset = train_data.create_tf_dataset_for_client(
        train_data.client_ids[0]
    )

print(type(example_dataset))
example_element = iter(example_dataset).next()
print(example_element)
# <class 'tensorflow.python.data.ops.dataset_ops.RepeatDataset'>
# {'age': <tf.Tensor: shape=(1,), dtype=int32, numpy=array([37], dtype=int32)>, 'workclass': <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'Local-gov'], dtype=object)>, ...    