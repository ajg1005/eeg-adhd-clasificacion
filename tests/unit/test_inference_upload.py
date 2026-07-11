from io import BytesIO
from types import SimpleNamespace

import pytest

from backend.inference.upload import ensure_csv_upload, read_csv_upload


def _upload(filename: str, content: bytes = b"a,b\n1,2\n"):
    return SimpleNamespace(filename=filename, file=BytesIO(content))


def test_ensure_csv_upload_accepts_csv_extension():
    ensure_csv_upload(_upload("patient.CSV"))


def test_ensure_csv_upload_rejects_non_csv_extension():
    with pytest.raises(ValueError, match="Solo se admiten archivos CSV"):
        ensure_csv_upload(_upload("patient.txt"))


def test_read_csv_upload_resets_file_pointer_and_reads_dataframe():
    file = _upload("dataset.csv")
    file.file.read()

    df = read_csv_upload(file)

    assert list(df.columns) == ["a", "b"]
    assert df.iloc[0].to_dict() == {"a": 1, "b": 2}
