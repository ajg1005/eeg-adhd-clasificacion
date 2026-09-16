from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from backend.datasets import service


def archive_bytes(files):
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, content in files:
            archive.writestr(name, content)
    return output.getvalue()


def test_zip_preserves_csv_bytes_and_uses_basename():
    csv = b"ID,Class\ns1,ADHD\n"
    archive = archive_bytes([("../../dataset.csv", csv)])
    assert service.unpack_training_upload(archive, "data.ZIP") == (csv, "dataset.csv")


@pytest.mark.parametrize("files", [[], [("x.txt", b"x")], [("a.csv", b"x"), ("b.csv", b"y")]])
def test_zip_requires_exactly_one_csv(files):
    with pytest.raises(ValueError, match="unico archivo CSV"):
        service.unpack_training_upload(archive_bytes(files), "data.zip")


def test_zip_rejects_oversized_csv(monkeypatch):
    monkeypatch.setattr(service, "MAX_ZIP_CSV_BYTES", 8)
    with pytest.raises(ValueError, match="limite"):
        service.unpack_training_upload(archive_bytes([("data.csv", b"x" * 9)]), "data.zip")


def test_zip_rejects_corrupt_archive():
    with pytest.raises(ValueError, match="No se pudo leer el ZIP"):
        service.unpack_training_upload(b"not a zip", "data.zip")
