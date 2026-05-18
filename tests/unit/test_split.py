"""Tests del modulo de splits cross-subject.

Estos tests defienden la metodologia central del TFG: que ningun paciente
aparezca a la vez en train y test (cross-subject CV) y que se rechacen
datasets que no se puedan estratificar.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.split import make_group_kfold_splits, make_group_shuffle_split


# Construye un dataset sintetico: 10 sujetos balanceados con N muestras cada uno
def _toy_dataset(n_subjects_per_class=5, samples_per_subject=10):
    rows = []
    groups = []
    labels = []
    subject_id = 0
    for label in (0, 1):
        for _ in range(n_subjects_per_class):
            subject_id += 1
            for _ in range(samples_per_subject):
                rows.append([float(subject_id) * 0.1, float(label) * 0.5])
                groups.append(f"s{subject_id}")
                labels.append(label)
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups, dtype=str)
    return X, y, g


# comprueba que ningun paciente aparece a la vez en train y test
def test_make_group_shuffle_split_no_subject_overlap():
    X, y, groups = _toy_dataset()
    X_train, X_test, y_train, y_test, groups_train, groups_test = make_group_shuffle_split(
        X, y, groups, test_size=0.2, random_state=42,
    )
    train_subjects = set(groups_train)
    test_subjects = set(groups_test)
    assert train_subjects.isdisjoint(test_subjects), (
        "Cross-subject violation: hay pacientes solapados entre train y test"
    )
    assert len(train_subjects) > 0
    assert len(test_subjects) > 0


# comprueba que cada paciente mantiene su etiqueta tras el split
def test_make_group_shuffle_split_preserves_label_consistency():
    X, y, groups = _toy_dataset()
    original_labels = dict(zip(groups, y))

    X_train, X_test, y_train, y_test, groups_train, groups_test = make_group_shuffle_split(
        X, y, groups, test_size=0.2, random_state=42,
    )

    for arr_groups, arr_labels in [(groups_train, y_train), (groups_test, y_test)]:
        for sid, label in zip(arr_groups, np.asarray(arr_labels)):
            assert int(label) == original_labels[sid]


# comprueba que se rechaza un dataset donde un sujeto tiene dos etiquetas distintas
def test_make_group_shuffle_split_rejects_inconsistent_subject_labels():
    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4]})
    y = np.array([0, 1, 0, 1])
    groups = np.array(["s1", "s1", "s2", "s2"])  # s1 tiene dos labels

    with pytest.raises(ValueError, match="una etiqueta"):
        make_group_shuffle_split(X, y, groups, test_size=0.5, random_state=42)


# comprueba que en cada fold del K-Fold no se solapan pacientes train/test
def test_make_group_kfold_splits_no_subject_overlap_per_fold():
    X, y, groups = _toy_dataset()
    splits = make_group_kfold_splits(X, y, groups, n_splits=5)

    assert len(splits) == 5

    for split in splits:
        train_subjects = set(split["groups_train"])
        test_subjects = set(split["groups_test"])
        assert train_subjects.isdisjoint(test_subjects), (
            f"Fold {split['fold']}: pacientes solapados train/test"
        )


# comprueba que cada paciente aparece como test exactamente una vez en los K folds
def test_make_group_kfold_splits_covers_all_subjects_in_test():
    X, y, groups = _toy_dataset()
    splits = make_group_kfold_splits(X, y, groups, n_splits=5)

    all_subjects = set(groups)
    seen_in_test = set()
    for split in splits:
        fold_test_subjects = set(split["groups_test"])
        overlap = seen_in_test & fold_test_subjects
        assert not overlap, f"Sujeto {overlap} aparece en test de dos folds"
        seen_in_test.update(fold_test_subjects)

    assert seen_in_test == all_subjects, (
        "No todos los sujetos aparecen como test en algun fold"
    )


# comprueba que con la misma seed se obtiene siempre el mismo split (reproducibilidad)
def test_make_group_kfold_splits_reproducible_with_fixed_seed():
    X, y, groups = _toy_dataset()
    splits_a = make_group_kfold_splits(X, y, groups, n_splits=5)
    splits_b = make_group_kfold_splits(X, y, groups, n_splits=5)

    for fa, fb in zip(splits_a, splits_b):
        assert list(fa["groups_test"]) == list(fb["groups_test"])
