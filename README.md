# Kompresja sygnałów EKG z wykorzystaniem autoenkodera

Ten projekt implementuje system do kompresji sygnałów elektrokardiograficznych (EKG) przy użyciu autoenkodera zbudowanego w TensorFlow i Keras. Głównym celem jest efektywna redukcja wymiarowości danych EKG przy zachowaniu kluczowych informacji diagnostycznych.

## 📦 Zbiór danych

Projekt wykorzystuje zbiór danych [ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) z platformy Kaggle, który obejmuje dane z baz MIT-BIH Arrhythmia oraz PTB Diagnostic ECG.

## 🏗️ Architektura

Model to symetryczny autoenkoder z następującą architekturą:

- **Enkoder**: `187 -> 100 -> 40 -> 20` (warstwa "bottleneck")
- **Dekoder**: `20 -> 40 -> 100 -> 187`

Zastosowano nieliniowe transformacje (`sqrt` przed enkoderem i `square` po dekoderze) w celu lepszego dopasowania do charakterystyki sygnałów EKG.

## 📊 Wyniki

- **Współczynnik kompresji**: **9.35:1** (redukcja z 187 do 20 cech)
- **Błąd rekonstrukcji (RMSE)**: `~0.0086` na zbiorze testowym

Wyniki pokazują, że model jest w stanie z dużą wiernością zrekonstruować sygnały, zachowując ich najważniejsze cechy morfologiczne.

## 🛠️ Technologie

- TensorFlow 2.x (z Keras API)
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
