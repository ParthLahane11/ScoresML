from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class Preprocess_several:
    def preprocess_pipeline(dataset = pd.DataFrame()):
        num_attributes = list(dataset.select_dtypes('int64').columns)
        cat_attributes = list(dataset.select_dtypes('object').columns)
        pipeline = ColumnTransformer([
            ('num', StandardScaler(), num_attributes),
            ('cat', OneHotEncoder(), cat_attributes),
        ])
        Processed_Data = pd.DataFrame(pipeline.fit_transform(dataset))
        return Processed_Data

    def separate_labels(Dataset = pd.DataFrame()):
        Dataset_labels = Dataset['G3'].to_numpy()
        Dataset_without_Labels = Dataset.drop(['G3'], axis=1)
        return Dataset_without_Labels, Dataset_labels

    def remove_uninteresting_labels(Dataset = pd.DataFrame(), correlations = {}):
        undesirable_Features = []
        for name, corr in correlations.items():
            if corr < 0.2 and corr > - 0.2 :
               undesirable_Features.append(name)
        Dataset_updated = Dataset.drop(undesirable_Features, axis = 1)
        return Dataset_updated