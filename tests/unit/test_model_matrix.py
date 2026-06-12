"""Tests for to_model_matrix (ueba.models.data_prep)."""

import numpy as np
import pytest

from ueba import config
from ueba.models.data_prep import to_model_matrix


def test_drops_exactly_the_non_feature_cols(ueba_df):
    X, cols = to_model_matrix(ueba_df)
    assert set(cols) == {"logon_count", "file_copy_count", "emails_sent", "http_upload_count"}
    assert not set(cols) & set(config.NON_FEATURE_COLS)
    assert X.shape == (len(ueba_df), 4)


def test_matrix_is_float32(ueba_df):
    X, _ = to_model_matrix(ueba_df)
    assert X.dtype == np.float32


def test_numeric_non_feature_cols_are_dropped(ueba_df):
    # role_sensitivity is numeric but belongs to the alert layer, never the model.
    assert "role_sensitivity" in config.NON_FEATURE_COLS
    _, cols = to_model_matrix(ueba_df)
    assert "role_sensitivity" not in cols


def test_leaked_string_column_fails_loudly(ueba_df):
    df = ueba_df.copy()
    df["future_ldap_attribute"] = "some-string"
    with pytest.raises(AssertionError, match="future_ldap_attribute"):
        to_model_matrix(df)
