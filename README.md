#penjelasan
**Bagian 1: Analisis Regresi**

1. **Import Library:**
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import altair as alt
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import r2_score
   ```
   - `pandas`: Untuk manipulasi dan analisis data.
   - `matplotlib.pyplot`: Untuk membuat visualisasi data.
   - `altair`:  Alternatif library visualisasi, menghasilkan grafik interaktif.
   - `sklearn.model_selection.train_test_split`: Untuk membagi data menjadi data latih dan data uji.
   - `sklearn.linear_model.LinearRegression`:  Untuk membuat model regresi linier.
   - `sklearn.metrics.r2_score`: Untuk menghitung nilai R-squared (koefisien determinasi) sebagai metrik evaluasi model.

2. **Membaca Data:**
   ```python
   df_corruptions = pd.read_csv('R04_corruptions.csv') 
   ```
   - Membaca data dari file CSV 'R04_corruptions.csv' ke dalam DataFrame pandas. Pastikan file CSV berada di direktori yang sama dengan file Python, atau berikan path lengkap ke file tersebut.

3. **Visualisasi Awal (Scatter Plot):**
   ```python
   chart = alt.Chart(df_corruptions).mark_point().encode(
       x='Corruption Perception (X)',
       y='Government Transparency (Y)',
       tooltip=['Corruption Perception (X)', 'Government Transparency (Y)']
   ).properties(
       title='Scatter Plot - Corruption Perception vs. Government Transparency'
   ).interactive()
   chart.save('corruption_scatter_plot.json')
   ```
   - Membuat scatter plot interaktif menggunakan `altair` untuk memvisualisasikan hubungan antara 'Corruption Perception (X)' dan 'Government Transparency (Y)'.
   - Grafik ini disimpan sebagai file JSON 'corruption_scatter_plot.json'. Kamu bisa membukanya di browser untuk melihat grafik interaktif.

4. **Statistik Deskriptif:**
   ```python
   print("\nStatistik Deskriptif:")
   print(df_corruptions.describe().to_markdown(numalign="left", stralign="left"))
   ```
   - Menampilkan statistik deskriptif (mean, standar deviasi, dll.) dari dataset.
   - `.to_markdown()` memformat output agar mudah dibaca.

5. **Membagi Data (Train-Test Split):**
   ```python
   X = df_corruptions[['Corruption Perception (X)']]  
   y = df_corruptions['Government Transparency (Y)']  
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   - `X`: Variabel independen (fitur), yaitu 'Corruption Perception (X)'.
   - `y`: Variabel dependen (target), yaitu 'Government Transparency (Y)'.
   - `train_test_split`: Membagi data menjadi data latih (`X_train`, `y_train`) dan data uji (`X_test`, `y_test`).
     - `test_size=0.2`: 20% data untuk pengujian.
     - `random_state=42`: Untuk memastikan pembagian data yang sama setiap kali kode dijalankan.

6. **Melatih Model Regresi Linier:**
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```
   - Membuat objek model `LinearRegression()`.
   - `model.fit()`: Melatih model menggunakan data latih (`X_train`, `y_train`).

7. **Membuat Prediksi:**
   ```python
   y_pred = model.predict(X_test)
   ```
   - `model.predict()`:  Membuat prediksi nilai 'Government Transparency (Y)' (`y_pred`) berdasarkan data uji (`X_test`).

8. **Evaluasi Model (R-squared):**
   ```python
   r2 = r2_score(y_test, y_pred)
   print(f'R-squared: {r2:.3f}')
   ```
   - `r2_score()`: Menghitung nilai R-squared untuk mengukur seberapa baik model cocok dengan data.

9. **Visualisasi Model:**
   ```python
   plt.scatter(X_train, y_train, color='blue', label='Data Latih')
   plt.scatter(X_test, y_test, color='green', label='Data Uji')
   plt.plot(X_test, y_pred, color='red', linewidth=2, label='Model Regresi')
   plt.xlabel('Corruption Perception (X)')
   plt.ylabel('Government Transparency (Y)')
   plt.title('Regresi Linier - Corruption Perception vs. Government Transparency')
   plt.legend()
   plt.savefig('linear_regression_model.png')
   plt.show()
   ```
   - Membuat scatter plot untuk memvisualisasikan:
     - Data latih (biru)
     - Data uji (hijau)
     - Garis regresi (merah)
   - Menyimpan plot sebagai file PNG 'linear_regression_model.png'.


**Bagian 2: Analisis Klasifikasi**

1. **Import Library:**
   ```python
   from sklearn.preprocessing import OneHotEncoder
   from sklearn.naive_bayes import GaussianNB
   from sklearn.metrics import accuracy_score
   ```
   - `sklearn.preprocessing.OneHotEncoder`: Untuk melakukan one-hot encoding pada variabel kategorikal.
   - `sklearn.naive_bayes.GaussianNB`:  Untuk membuat model Naive Bayes.
   - `sklearn.metrics.accuracy_score`: Untuk menghitung akurasi model klasifikasi.

2. **Membaca Data:**
   ```python
   df_bank_customers = pd.read_csv('K04_bank_customers.csv', delimiter=';')
   ```
   - Membaca data dari file CSV 'K04_bank_customers.csv'. 
   - `delimiter=';'`: Menentukan bahwa data dalam file CSV dipisahkan oleh titik koma.

3. **Statistik Deskriptif:**
   ```python
   print("\nStatistik Deskriptif:")
   print(df_bank_customers.describe().to_markdown(numalign="left", stralign="left"))
   ```
   -  Sama seperti di bagian regresi, menampilkan statistik deskriptif dari dataset.

4. **One-Hot Encoding:**
   ```python
   encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
   categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
   encoded_features = encoder.fit_transform(df_bank_customers[categorical_features])

   encoded_df = pd.DataFrame(encoded_features)
   encoded_df = encoded_df.add_prefix('encoded_')

   df_bank_customers_encoded = pd.concat([df_bank_customers, encoded_df], axis=1)
   df_bank_customers_encoded = df_bank_customers_encoded.drop(categorical_features, axis=1)
   ```
   - `OneHotEncoder`: Mengubah variabel kategorikal (seperti 'job', 'marital', 'education') menjadi representasi numerik yang dapat digunakan oleh model Naive Bayes.
     - `handle_unknown='ignore'`: Untuk menangani nilai kategori baru yang mungkin muncul di data yang belum pernah dilihat sebelumnya.
     - `sparse_output=False`:  Untuk menghasilkan array NumPy biasa, bukan matriks sparse.
   -  Kode ini membuat DataFrame baru (`encoded_df`) dengan fitur yang sudah di-encode, menggabungkannya dengan DataFrame asli, dan menghapus kolom kategorikal asli.

5. **Membagi Data (Train-Test Split):**
   ```python
   X = df_bank_customers_encoded.drop('y', axis=1)
   y = df_bank_customers_encoded['y']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   - `X`: Semua kolom kecuali kolom target 'y'.
   - `y`: Kolom target 'y'.
   - Data dibagi menjadi data latih dan data uji, sama seperti di bagian regresi.

6. **Melatih Model Naive Bayes:**
   ```python
   model = GaussianNB()
   model.fit(X_train, y_train)
   ```
   - Membuat objek model `GaussianNB()`.
   - `model.fit()`: Melatih model menggunakan data latih
