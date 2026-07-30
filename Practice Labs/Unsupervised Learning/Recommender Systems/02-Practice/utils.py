from sklearn.preprocessing import LabelEncoder

def column_encoder(dataset, *cols):
    encoders = {}
    
    for col in cols:
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])
        encoders[col] = encoder
        
    return dataset, encoders
