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