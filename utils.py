import pandas as pd
import pickle


def get_encoder(col='player'):
    """
    Get Label Encoder object for a given column
    Return Type - LabelEncoder object for the specified column
    """

    with open(f"encoders/{col}_encoder.pkl", "rb") as fp:
        encoder = pickle.load(fp)

    return encoder