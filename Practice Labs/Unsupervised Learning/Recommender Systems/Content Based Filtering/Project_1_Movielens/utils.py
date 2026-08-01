def multi_hot_encode(items, column, separator="|"):
    
    encoded_rows = []

    for value in column:
        item_row = value.split(separator)
        
        encoded_array = []
        
        for item in items:
            if item in item_row:
                encoded_array.append(1)
            else:
                encoded_array.append(0)
        encoded_rows.append(encoded_array)

    return encoded_rows