from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def make_group_shuffle_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # Crear el objeto que va a separar train y test por grupos
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    # Sacar los índices de train y test
    train_idx, test_idx = next(splitter.split(X, y, groups))

    # Separar las features
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    # Separar las etiquetas
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Separar los grupos o pacientes
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Devolver todo separado
    return X_train, X_test, y_train, y_test, groups_train, groups_test