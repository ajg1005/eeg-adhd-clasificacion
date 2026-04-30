from pathlib import Path
import argparse
import json

from inference import predict_eeg_file


def main():
    parser = argparse.ArgumentParser(
        description="Predicción ADHD/Control a partir de un archivo EEG en CSV."
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Ruta al archivo CSV EEG.",
    )

    args = parser.parse_args()

    result = predict_eeg_file(Path(args.file_path))

    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()