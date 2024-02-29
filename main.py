import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import warnings

# Menyembunyikan peringatan runtime (opsional)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Membaca dataset
dataset = pd.read_csv("dataset-test.csv", header=0, low_memory=False)

# Membersihkan nama kolom dari spasi di awal dan akhir
dataset.columns = [col.strip() for col in dataset.columns]

# Mengisi nilai yang hilang dengan mean
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(dataset.drop(['Label', 'Source IP', 'Flow ID', 'Destination IP', 'Timestamp'], axis=1)),
                 columns=dataset.drop(['Label', 'Source IP', 'Flow ID', 'Destination IP', 'Timestamp'], axis=1).columns)

# Pisahkan fitur dan target
y = dataset['Label']

# Identifikasi dan hapus fitur konstan
constant_features = X.columns[X.nunique() == 1]
if len(constant_features) > 0:
    print("Fitur berikut dihapus karena konstan:")
    print(", ".join(constant_features))
    X = X.drop(columns=constant_features)

# Pisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Metode 1: K-Nearest Neighbor (KKN)
kkn_model = KNeighborsClassifier(n_neighbors=3)
kkn_model.fit(X_train, y_train)
kkn_predictions = kkn_model.predict(X_test)

# Metode 2: Random Forest (RF)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluasi performa sebelum pemilihan fitur
print("\nEvaluasi Akurasi Sebelum Pemilihan Fitur:")
print("Akurasi KKN:", accuracy_score(y_test, kkn_predictions))
print("Akurasi RF:", accuracy_score(y_test, rf_predictions))

# Pemilihan fitur menggunakan ANOVA (Contoh, bisa disesuaikan dengan metode lain)
k_best = SelectKBest(score_func=f_classif, k=4)  # Gantilah 10 dengan jumlah fitur yang ingin dipilih
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)

# Latih model pada fitur terpilih
kkn_model.fit(X_train_selected, y_train)
rf_model.fit(X_train_selected, y_train)

# Lakukan prediksi pada fitur terpilih
kkn_predictions_selected = kkn_model.predict(X_test_selected)
rf_predictions_selected = rf_model.predict(X_test_selected)

# Evaluasi performa setelah pemilihan fitur
print("\nEvaluasi Akurasi Setelah Pemilihan Fitur:")
print("Akurasi KKN:", accuracy_score(y_test, kkn_predictions_selected))
print("Akurasi RF:", accuracy_score(y_test, rf_predictions_selected))

# Membuat output tabel
accuracy_kkn_before = accuracy_score(y_test, kkn_predictions)
accuracy_rf_before = accuracy_score(y_test, rf_predictions)
accuracy_kkn_after = accuracy_score(y_test, kkn_predictions_selected)
accuracy_rf_after = accuracy_score(y_test, rf_predictions_selected)

data = {
    'Metode': ['K-Nearest Neighbor (KKN)', 'Random Forest (RF)'],
    'Sebelum Pemilihan Fitur': [accuracy_kkn_before, accuracy_rf_before],
    'Setelah Pemilihan Fitur': [accuracy_kkn_after, accuracy_rf_after]
}

output_table = pd.DataFrame(data)

# Menampilkan tabel output tanpa indeks baris
print("\nTabel Output Akurasi:")
print(output_table.to_string(index=False))
