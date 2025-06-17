import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import kagglehub

# Funkcje pomocnicze do plików
def file_exists(filepath: str) -> bool:
    return Path(filepath).exists()

def delete_file(filepath: str):
    try:
        Path(filepath).unlink()
    except FileNotFoundError:
        pass

def run_shell(command: str) -> int:
    return os.system(command)

# Ładowanie danych - pobieranie lub lokalnie
def load_ecg_dataset():
    try:
        dataset_path = Path(kagglehub.dataset_download("shayanfazeli/heartbeat"))
        print(f"Dataset path: {dataset_path}")
        
        train_data = pd.read_csv(dataset_path / 'mitbih_train.csv', header=None)
        test_data = pd.read_csv(dataset_path / 'mitbih_test.csv', header=None)
        abnormal_data = pd.read_csv(dataset_path / 'ptbdb_abnormal.csv', header=None)
        normal_data = pd.read_csv(dataset_path / 'ptbdb_normal.csv', header=None)
    except Exception as err:
        print(f"Pobieranie nie powiodło się: {err}")
        print("Przechodzę do lokalnych plików lub rozpakowywania 7z...")
        
        # Jeśli pliki nie istnieją, próbujemy rozpakować
        if not file_exists("mitbih_test.csv"):
            success = any(run_shell(cmd) == 0 for cmd in ["7za e data.7z.001", "7z e data.7z.001", "7zz e data.7z.001"])
            if not success:
                print("Zainstaluj 7za lub 7z, by rozpakować archiwum.")
        
        train_data = pd.read_csv('mitbih_train.csv', header=None)
        test_data = pd.read_csv('mitbih_test.csv', header=None)
        abnormal_data = pd.read_csv('ptbdb_abnormal.csv', header=None)
        normal_data = pd.read_csv('ptbdb_normal.csv', header=None)
    return train_data, test_data, abnormal_data, normal_data

# Przygotowanie danych do treningu/testu
def prepare_data(df: pd.DataFrame, label_value: int):
    subset = df[df[187] == label_value]
    features = subset.iloc[:, :187].values.astype(np.float32)
    labels = subset[187].values.tolist()
    return features, labels

# Wizualizacja sygnałów EKG
def plot_ecg_samples(normal_signals, abnormal_signals):
    plt.figure(figsize=(20,10))
    for idx in range(8):
        plt.subplot(4, 4, idx+1)
        plt.plot(normal_signals[idx])
        plt.title(f"Normal ECG Sample {idx+1}")
        plt.xlabel("Time")
    for idx in range(8, 16):
        plt.subplot(4, 4, idx+1)
        plt.plot(abnormal_signals[idx])
        plt.title(f"Abnormal ECG Sample {idx-7}")
        plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig('ecg_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

# Autoenkoder do kompresji sygnału EKG
class ECGAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(ECGAutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(188,)),
            tf.keras.layers.Dense(40, activation='relu'),
            tf.keras.layers.Dense(20, activation='linear')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(40, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(188, activation='sigmoid')
        ])
    def call(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded
    def encode(self, x):
        return self.encoder(tf.sqrt(x))
    def decode(self, encoded):
        return tf.square(self.decoder(encoded))

# Normalizacja danych
def normalize(train_arr, test_arr):
    min_val = np.min(train_arr, axis=(0,1))
    max_val = np.max(train_arr, axis=(0,1))
    train_norm = (train_arr - min_val) / (max_val - min_val)
    test_norm = (test_arr - min_val) / (max_val - min_val)
    return train_norm, test_norm

# Pojedyncza epoka treningowa
def train_epoch(model, optimizer, dataset, val_data, loss_fn):
    train_ds = tf.data.Dataset.from_tensor_slices(dataset).shuffle(1000).batch(250)
    for batch in train_ds:
        with tf.GradientTape() as tape:
            recon = model(batch)
            loss = loss_fn(batch, recon)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    val_recon = model(val_data)
    val_loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(val_recon - val_data), axis=1)))
    return float(val_loss)

# Rysowanie błędu trenowania
def plot_training_errors(epochs, errors):
    plt.figure()
    plt.plot(epochs, errors)
    plt.title("Error During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.savefig('training_error.png', dpi=300, bbox_inches='tight')
    plt.close()

# Główna funkcja
def main():
    train_df, test_df, abn_df, norm_df = load_ecg_dataset()
    
    x_train, y_train = prepare_data(train_df, label_value=0)  # Normal train
    x_test, y_test = prepare_data(test_df, label_value=0)    # Normal test
    
    plot_ecg_samples(x_test, prepare_data(test_df, label_value=1)[0])  # normal + abnormal test samples
    
    train_norm, test_norm = normalize(train_df.values.astype(np.float32), test_df.values.astype(np.float32))
    
    model = ECGAutoEncoder()
    optimizer = tf.keras.optimizers.Adam(1e-3)
    loss_function = tf.keras.losses.MeanSquaredError()
    
    epochs = []
    errors = []
    error = 1.0
    no_improve_count = 0
    best_weights = None
    epoch = 1
    
    while error > 8.6e-3 and epoch < 200 and no_improve_count < 15:
        val_error = train_epoch(model, optimizer, train_norm, test_norm, loss_function)
        diff = error - val_error
        if diff < 1e-6:
            no_improve_count += 1
        else:
            no_improve_count = 0
            best_weights = model.get_weights()
        error = val_error
        epochs.append(epoch)
        errors.append(error)
        print(f"Epoch {epoch} - Validation Error: {error:.6f} Diff: {diff:.6f}")
        epoch += 1
    
    if best_weights:
        model.set_weights(best_weights)
    
    if len(epochs) > 1:
        plot_training_errors(epochs, errors)
    else:
        print("Uczenie zbyt szybkie, brak wykresu")
    
    # Ocena końcowa
    compressed_train = model.encode(train_norm)
    compressed_test = model.encode(test_norm)
    decompressed_train = model.decode(compressed_train)
    decompressed_test = model.decode(compressed_test)
    
    train_rmse = float(tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(train_norm - decompressed_train), axis=1))))
    test_rmse = float(tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(test_norm - decompressed_test), axis=1))))
    
    print(f"Końcowy błąd treningowy (RMSE): {train_rmse:.6f}")
    print(f"Końcowy błąd testowy (RMSE): {test_rmse:.6f}")
    
    plt.figure()
    plt.plot(test_norm[0], label='Oryginalny sygnał')
    plt.plot(decompressed_test[0].numpy(), label='Odtworzony sygnał')
    plt.title("Porównanie oryginału i rekonstrukcji")
    plt.legend()
    plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    train_err_flat = (decompressed_train - train_norm).numpy().flatten()
    test_err_flat = (decompressed_test - test_norm).numpy().flatten()
    
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.hist(train_err_flat, bins=25, log=True)
    plt.title("Rozkład błędów treningowych")
    plt.subplot(1, 2, 2)
    plt.hist(test_err_flat, bins=25, log=True)
    plt.title("Rozkład błędów testowych")
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sprzątanie lokalnych plików, jeśli potrzebne
    try:
        if 'dataset_path' not in locals():
            for f in ["mitbih_test.csv", "mitbih_train.csv", "ptbdb_abnormal.csv", "ptbdb_normal.csv"]:
                delete_file(f)
        else:
            print("Dane pobrane z Kaggle, brak konieczności sprzątania.")
    except Exception:
        pass

if __name__ == "__main__":
    main()
