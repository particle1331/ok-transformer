import pandas as pd


def prepare_features(input):
    """Prepare features for dict vectorizer."""

    X = pd.DataFrame(input)
    X['PU_DO'] = X['PULocationID'].astype(str) + '_' + X['DOLocationID'].astype(str)
    X = X[['PU_DO', 'trip_distance']].to_dict(orient='records')
    
    return X