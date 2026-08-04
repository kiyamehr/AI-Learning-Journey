from sklearn.preprocessing import StandardScaler

def scale_numeric_features(*cols):

    """
    Scale the numeric columns of one or more pandas DataFrames using
    a single StandardScaler.

    The scaler is fitted on the first DataFrame provided and then used
    to transform the remaining DataFrames. This prevents data leakage
    by ensuring that statistics (mean and standard deviation) are
    learned only from the training data.

    Only columns with dtype ``int64`` or ``float64`` are scaled.
    Boolean and categorical columns are left unchanged.

    Parameters
    ----------
    *cols : pandas.DataFrame
        One or more DataFrames to scale. The first DataFrame is used
        to fit the scaler, while all subsequent DataFrames are only
        transformed.

    Returns
    -------
    None
        The input DataFrames are modified in place.
    """
    
    
    scaler = StandardScaler()

    for i, col in enumerate(cols):
        numeric_cols = col.select_dtypes(['int64', 'float64']).columns
        
        if i == 0:
            col[numeric_cols] = scaler.fit_transform(col[numeric_cols])
        else:
            col[numeric_cols] = scaler.transform(col[numeric_cols])

def multi_hot_encode(items, column, separator="|"):
    """
    Convert a column containing multiple items into multi-hot encoded vectors.

    Parameters:
        items (list):
            A list of all possible items/categories.
            Example: ["Action", "Comedy", "Drama"]

        column (iterable):
            A column where each value contains one or more items
            separated by the given separator.
            Example: ["Action|Comedy", "Drama", "Action|Drama"]

        separator (str):
            The character used to separate multiple items in each value.
            Defaults to "|".

    Returns:
        list:
            A list of multi-hot encoded vectors, where each item is represented
            by 1 if it is present and 0 if it is absent.
    """

    # Store the encoded vector for each row
    encoded_rows = []

    # Loop through every value in the input column
    for value in column:

        # Split the value into individual items using the separator
        # Example: "Action|Comedy" -> ["Action", "Comedy"]
        item_row = value.split(separator)

        # Store the encoded values for the current row
        encoded_array = []

        # Check every possible item
        for item in items:

            # If the item exists in the current row, encode it as 1
            if item in item_row:
                encoded_array.append(1)

            # Otherwise, encode it as 0
            else:
                encoded_array.append(0)

        # Add the encoded vector for this row to the results
        encoded_rows.append(encoded_array)

    # Return the multi-hot encoded data
    return encoded_rows
