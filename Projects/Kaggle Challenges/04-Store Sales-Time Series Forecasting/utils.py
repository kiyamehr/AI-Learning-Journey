import pandas as pd

def introduce_data(datasets):
    for name, df in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"{name} Dataset")
        print(f"{'=' * 50}")
        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nFirst 5 rows:")
        print(df.head())
        
def preprocess(datasets):
    
    df_train = datasets["Train"]
    df_test = datasets["Test"]
    df_holidays = datasets["Holidays"]
    df_oil = datasets["Oil"]
    df_stores = datasets["Stores"]
    df_transactions = datasets["Transactions"]
    
    for name, df in datasets.items():
        # Gaurd Clause
        if 'date' not in df.columns:
            continue
        
        # To datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # adding date columns
        if name in ['Train', 'Test']:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['dayofyear'] = df['date'].dt.dayofyear
    # merging useful columns:
    
    df_train = df_train.merge(df_stores, on='store_nbr', how='left')
    df_test = df_test.merge(df_stores, on='store_nbr', how='left')
    
    df_train = df_train.merge(df_oil, on='date', how='left')
    df_test = df_test.merge(df_oil, on='date', how='left')
    
    df_train = df_train.merge(df_holidays, on='date', how='left')
    df_test = df_test.merge(df_holidays, on='date', how='left')
            
    # Dropping unused columns
    # df_train = df_train.drop(['id', 'date'], axis=1)
    # df_test = df_test.drop(['id', 'date'], axis=1)
    
    # Renaming duplicate columns:
    df_train = df_train.rename(columns={
        'type_x' : 'store_type',
        'type_y' : 'holiday_type',
    })
    
    df_test = df_test.rename(columns={
        'type_x' : 'store_type',
        'type_y' : 'holiday_type',
    })
    
    # Converting transfered to int
    df_train['transferred'] = df_train['transferred'].notna().astype(int)
    df_train['transferred'] = df_train['transferred'].fillna(False).astype(int)
    
    df_test['transferred'] = df_test['transferred'].notna().astype(int)
    df_test['transferred'] = df_test['transferred'].fillna(False).astype(int)
    
    # Seperating df_train, y_train
    # y_train = df_train['sales']
    # df_train = df_train.drop('sales', axis=1)
    
    # categorical_cols = df_train.select_dtypes('str').columns
    
    # col_transformer = ColumnTransformer([
    #     ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    # ], remainder='passthrough')
    
    # df_train_encoded = col_transformer.fit_transform(df_train)
    # df_test_encoded = col_transformer.transform(df_test)
    
    return df_train, df_test
