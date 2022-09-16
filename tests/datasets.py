from sklearn.datasets import load_diabetes, load_iris, fetch_openml


class IrisDataset:
    dataset = load_iris(as_frame=True)
    target_names = dataset["target_names"]
    target_count = len(dataset["target_names"])
    X = dataset["data"]
    y = dataset["target"].astype("category")
    
    model_feature_id = "target"
    

class DiabetesDataset:
    dataset = load_diabetes(as_frame=True)
    X = dataset["data"]
    y = dataset["target"]
    model_feature_id = "target"


class TitanicDataset:
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    model_feature_id = "survived"
