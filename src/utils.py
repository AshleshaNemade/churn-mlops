import pandas as pd

def transform_input(data, features, scaler, numerical_cols):
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Step 1: One-hot encode (same as training)
    df = pd.get_dummies(df)

    # Step 2: Align with training features
    df = df.reindex(columns=features, fill_value=0)

    # Step 3: Scale ONLY numerical columns
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df