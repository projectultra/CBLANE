import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
integer_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(categories = [[0,1,2,3]])

def one_hot_encode(sequences,labels):
  main_labels = []
  input_features = []
  idx=0
  for sequence,label in zip(sequences,labels):
    integer_encoded = integer_encoder.fit_transform(list(sequence))
    integer_encoded = np.array(integer_encoded).reshape(-1, 1)
    try:
      one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
      input_features.append(one_hot_encoded.toarray())
      main_labels.append(label)
      del one_hot_encoded
      del integer_encoded
    except ValueError:
      print(idx)
      pass
    idx+=1
  return np.stack(input_features),np.array(main_labels)

def encode_raw_data():
    # Open the file for reading
    sequences=[]
    labels=[]
    with open('train.data', 'r') as file:
        for line in file:
            # Split each line by whitespace into sequence and label
            parts = line.strip().split()
            sequences.append(parts[1])
            labels.append(int(parts[2]))
        train_features,train_labels = one_hot_encode(sequences,labels)
        
    # Open the file for reading
    sequences=[]
    labels=[]
    with open('test.data', 'r') as file:
        for line in file:
            # Split each line by whitespace into sequence and label
            parts = line.strip().split()
            sequences.append(parts[1])
            labels.append(int(parts[2]))
        test_features,test_labels = one_hot_encode(sequences,labels)

    # Open the file for reading
    sequences=[]
    labels=[]
    with open('validation.data', 'r') as file:
        for line in file:
            # Split each line by whitespace into sequence and label
            parts = line.strip().split()
            sequences.append(parts[1])
            labels.append(int(parts[2]))
        validation_features,validation_labels = one_hot_encode(sequences,labels)