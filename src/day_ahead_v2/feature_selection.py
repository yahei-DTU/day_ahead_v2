def feature_selection(df, target_col, threshold=0.1):
    """
    Perform feature selection based on correlation with the target variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features and target variable.
    target_col (str): The name of the target column.
    threshold (float): The minimum absolute correlation value to keep a feature.

    Returns:
    pd.DataFrame: A DataFrame containing only the selected features and the target variable.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Get the absolute correlation values with the target variable
    target_corr = corr_matrix[target_col].abs()

    # Select features that have a correlation above the threshold
    selected_features = target_corr[target_corr > threshold].index.tolist()

    # Ensure the target column is included in the selected features
    if target_col not in selected_features:
        selected_features.append(target_col)

    return df[selected_features]
