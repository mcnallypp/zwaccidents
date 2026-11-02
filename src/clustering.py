
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_accidents(df, radius_km=5, min_samples=3):
    """
    Assign spatial cluster labels to accident records.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain Lat and Long columns.
    radius_km : float
        Neighbourhood radius in kilometres.
    min_samples : int
        Minimum number of nearby observations for a cluster.

    Returns
    -------
    pandas.DataFrame
        Copy of the input data with a Cluster column.
        Cluster -1 represents unclustered observations.
    """

    if radius_km <= 0:
        raise ValueError("radius_km must be greater than zero.")

    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")

    result = df.copy()

    result = result.dropna(subset=["Lat", "Long"]).copy()

    coordinates = np.radians(
        result[["Lat", "Long"]].to_numpy()
    )

    earth_radius_km = 6371.0088

    model = DBSCAN(
        eps=radius_km / earth_radius_km,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree"
    )

    result["Cluster"] = model.fit_predict(coordinates)

    return result
