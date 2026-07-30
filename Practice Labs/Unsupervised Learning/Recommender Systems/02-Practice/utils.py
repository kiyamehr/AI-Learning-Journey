from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def column_encoder(dataset, *cols):
    encoders = {}
    
    for col in cols:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])
        encoders[col] = encoder
        
    return dataset, encoders

def create_tf_dataset(data, target, *cols):
    features = {
        col: data[col].values for col in cols
    }
    
    labels = data[target].values
    
    tf_ds = tf.data.Dataset.from_tensor_slices((features, labels))
    
    return tf_ds