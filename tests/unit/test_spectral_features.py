
import numpy as np
from scripts.spectral_features import extract_spectral_features

# comprueba que se genera una potencia absoluta por banda para cada canal
def test_extract_spectral_features_returns_band_powers_per_channel():
    # Crea un array X_epochs sintetico (n_epochs, n_samples, n_channels)

    x_epochs = x_epochs = np.random.default_rng(42).standard_normal((2, 256, 3))

    channels =   channels = ["Fp1", "Fp2", "F3"]

    features = extract_spectral_features(x_epochs, channels, sfreq=128, nperseg=128)

    #Comprueba que para cada canal aparece una columna por cada banda
    for channel in channels:
        for band in ["delta", "theta", "alpha","beta","gamma"]:
            column = f"{channel}_{band}_abs_power"
            assert column in features.columns