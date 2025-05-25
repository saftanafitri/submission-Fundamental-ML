
# Laporan Proyek Machine Learning - Saftana Fitri

## Domain Proyek

**Deteksi Dini Depresi pada Siswa Menggunakan Algoritma Machine Learning**

Depresi pada remaja, khususnya siswa, merupakan salah satu gangguan kesehatan mental yang signifikan dan semakin mendapat perhatian. Menurut data WHO, sekitar 14% (1 dari 7) remaja berusia 10–19 tahun mengalami gangguan kesehatan mental, dengan depresi dan kecemasan sebagai penyebab utama yang berdampak pada prestasi belajar dan interaksi sosial mereka [1].

Masalah ini perlu segera ditangani karena:
- **Tingginya prevalensi** depresi di kalangan pelajar.
- **Minimnya tenaga profesional** kesehatan mental di sekolah-sekolah.
- **Stigma sosial** terhadap penderita depresi membuat mereka enggan untuk berkonsultasi.

Machine learning menawarkan pendekatan alternatif untuk membantu proses deteksi awal depresi berdasarkan data.

**Referensi:**
[1] World Health Organization. (2021). Adolescent mental health. https://www.who.int/news-room/fact-sheets/detail/adolescent-mental-health

## Business Understanding

### Problem Statements

- Bagaimana mengklasifikasikan apakah seorang siswa mengalami depresi berdasarkan data?
- Apa model machine learning yang paling efektif dalam mendeteksi depresi tidaknya siswa tersebut berdasarkan fitur-fitur yang tersedia?

### Goals

- Membangun model klasifikasi untuk mendeteksi siswa yang mengalami depresi.
- Membandingkan performa beberapa algoritma machine learning untuk memilih model terbaik.

### Solution Statements

- Menerapkan beberapa algoritma klasifikasi seperti Logistic Regression, SVM, dan Random Forest.
- Melakukan eksplorasi dan preprocessing data untuk meningkatkan akurasi model.
- Menggunakan metrik akurasi, precision, recall, dan F1-score untuk mengevaluasi performa setiap model.

## Data Understanding

Dataset diperoleh dari [Kaggle](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset).

Dataset ini mengumpulkan berbagai informasi yang bertujuan untuk memahami, menganalisis, dan memprediksi tingkat depresi di kalangan siswa. Dataset ini dirancang untuk keperluan penelitian di bidang psikologi, ilmu data, dan pendidikan, dengan memberikan wawasan mengenai faktor-faktor yang berkontribusi terhadap tantangan kesehatan mental siswa. Selain itu, dataset ini juga membantu dalam merancang strategi intervensi dini yang efektif untuk mendukung kesehatan mental siswa.

Jumlah data: 27901 baris  
Target: `DEPRESSION` (0 = tidak depresi, 1 = depresi)

**Fitur pada dataset meliputi:**
- ID: Unique identifier for each student
- Demographics: Age, Gender, City
- Academic Indicators: CGPA, Academic Pressure, Study Satisfaction 
- Lifestyle & Wellbeing: Sleep Duration, Dietary Habits, Work Pressure Job Satisfaction, Work/Study Hours
- Additional Factors: Profession, Degree, Financial Stress, Family History of Mental Illness, and whether the student has ever had suicidal thoughts 

### EDA (Exploratory Data Analysis)

Tidak ada Duplikasi dan Nilai NULL pada data.

1. Distribusi Data Tiap Fitur (Histogram)
- id: Distribusi merata, bukan fitur informatif (perlu dihapus).
- Academic Pressure: Terdistribusi bervariasi, persepsi tekanan akademik sangat beragam.
- Work Pressure: Sangat tidak seimbang (hampir semua bernilai 0).
- CGPA: Skor IPK bervariasi, mayoritas > 6.0.
- Study Satisfaction: Relatif merata, dengan sedikit dominasi pada skor tinggi (4 sampai 5).
- Job Satisfaction: Mayoritas bernilai rendah atau nol (kurang)informatif.
- Work/Study Hours: Variasi tinggi, sebagian besar siswa belajar hingga 12 jam/hari.
- Depression: Distribusi biner 0 dan 1.

2. Distribusi Kelas Target
- Depression = 1 (mengalami depresi): ~16.000 siswa.
- Depression = 0 (tidak mengalami depresi): ~11.500 siswa.
Dataset sedikit tidak seimbang, namun masih dalam batas toleransi.

3. Korelasi Antar Fitur (Heatmap)
Korelasi terhadap Depression:
- Academic Pressure: 0.47 (positif kuat)
- Work/Study Hours: 0.21 (positif sedang)
- Age: -0.23 (negatif sedang)
- Study Satisfaction: -0.17 (negatif lemah ke sedang)

4. Skewness Kolom
Kolom dengan skewness ekstrem (di luar rentang -0.5 hingga 0.5):
- Work Pressure: 108.59
- Job Satisfaction: 74.11
Interpretasi:
Kedua fitur ini sangat tidak seimbang (sangat banyak nilai 0 atau 1), kemungkinan besar tidak memberi kontribusi signifikan terhadap model.

## Data Preparation

Tahapan data preparation meliputi:
- **Menghapus fitur-fitur (kolom) yang tidak relevan**: Variabel seperti `id`, `City`, `Profession`, `Job Satisfaction`, `Work Pressure` dihapus karena tidak memiliki kontribusi berarti dan tidak relevan terhadap target.
- **Check Outliers**: Outlier tidak ditangani karena jumlahnya sangat kecil kurang dari 0.05% dari total data untuk tiap fitur sehingga dampaknya terhadap model cenderung minimal.
- **Encoding Fitur Kategori**:mengubah fitur kategorikal menjadi bentuk numerik agar dapat diproses oleh algoritma machine learning. Label Encoding digunakan untuk fitur kategorikal biner seperti `Gender`, `Have you ever had suicidal thoughts ?`, dan `Family History of Mental Illness`. One-Hot Encoding digunakan untuk fitur kategorikal dengan lebih dari dua kategori seperti `Sleep Duration`, `Dietary Habits`, `Degree`, dan `Financial Stress`.
- **Train-Test-Split**: Data dibagi menjadi data latih dan data uji menggunakan train_test_split. Sebanyak 70% data digunakan untuk melatih model, dan 30% sisanya digunakan untuk menguji performa model. 
- **Standarisasi**: Fitur dinormalisasi menggunakan `StandardScaler` untuk memastikan bahwa semua fitur berada dalam skala yang sama. Banyak algoritma machine learning (seperti Logistic Regression dan SVM) sensitif terhadap skala data, sehingga fitur dengan skala besar dapat mendominasi hasil prediksi jika tidak dilakukan scaling. Hasilnya, data memiliki distribusi dengan rata-rata 0 dan standar deviasi 1.

## Modeling

Beberapa algoritma dikembangkan dan dievaluasi:
- **Logistic Regression**
Logistic Regression adalah algoritma yang sederhana, cepat, dan mudah diinterpretasikan. Kelebihan utamanya terletak pada kemampuan memberikan pemahaman yang jelas terhadap pengaruh masing-masing fitur terhadap probabilitas depresi, menjadikannya sangat cocok sebagai baseline model atau untuk kebutuhan yang menekankan interpretabilitas. Namun, Logistic Regression memiliki keterbatasan karena hanya bisa memodelkan hubungan linear antara variabel input dan target. Ia juga sensitif terhadap multikolinearitas dan outlier. Meskipun demikian, pada kasus ini, model ini justru menunjukkan performa yang sangat baik, dengan recall dan f1-score tertinggi pada kelas siswa yang mengalami depresi, menjadikannya pilihan yang baik terutama dalam konteks pencegahan dini.
- **Random Forest**
Random Forest adalah model yang lebih kompleks dan tangguh, sangat efektif dalam menangani data dengan hubungan non-linear dan interaksi antar fitur yang kompleks. Model ini bekerja dengan membangun banyak decision tree dan melakukan rata-rata untuk mengurangi overfitting, sehingga memberikan performa yang stabil dan akurat. Kelebihan lain dari Random Forest adalah kemampuannya dalam menentukan feature importance, sehingga berguna untuk mengetahui faktor-faktor dominan penyebab depresi. Meski demikian, Random Forest memerlukan waktu pelatihan dan sumber daya yang lebih besar, serta tidak seinterpretatif Logistic Regression. Namun, untuk kasus deteksi depresi siswa yang variabel-variabelnya saling terkait secara kompleks (seperti tekanan akademik, jam belajar, dan kepuasan belajar), Random Forest menjadi pilihan yang sangat tepat dari segi prediktif.
- **Support Vector Machine (SVM)**
Support Vector Machine (SVM) adalah algoritma yang sangat efektif untuk data berdimensi tinggi dan mampu menangani klasifikasi non-linear menggunakan kernel trick. SVM dikenal mampu membentuk margin pemisah optimal antar kelas. Namun, kekurangan SVM terletak pada waktu pelatihan yang lambat terutama untuk dataset besar, dan sulitnya interpretasi model,terutama saat menggunakan kernel non-linear. Dalam konteks proyek ini, performa SVM sebanding dengan Random Forest, tetapi tidak memberikan keunggulan signifikan, sehingga lebih cocok digunakan untuk eksperimen tambahan atau sebagai model cadangan dalam validasi silang.

**Model terbaik dalam proyek ini adalah Logistic Regression. Model ini dipilih karena memberikan performa terbaik pada kelas depresi dengan recall sebesar 0.88 dan f1-score tertinggi, yang sangat penting untuk meminimalkan kasus depresi yang tidak terdeteksi. Selain itu, Logistic Regression juga mudah diinterpretasikan, ringan secara komputasi, dan cocok digunakan dalam konteks sosial seperti ini, di mana pemahaman terhadap faktor risiko sangat dibutuhkan oleh pihak sekolah atau konselor. Dibandingkan dengan Random Forest dan SVM, model ini lebih praktis dan efisien untuk implementasi nyata.**

## Evaluation

Untuk mengevaluasi kinerja model dalam mendeteksi apakah seorang siswa mengalami depresi atau tidak, digunakan empat metrik utama yang umum dalam klasifikasi biner, yaitu:
- **Accuracy**: Mengukur proporsi prediksi yang benar terhadap seluruh prediksi.
- **Precision**: Mengukur seberapa tepat model dalam memprediksi positif (siswa yang depresi).
- **Recall**: Mengukur seberapa banyak kasus positif yang berhasil ditangkap oleh model.
- **F1-Score**: Harmonik dari precision dan recall.

### Hasil Evaluasi
|-----------------------|----------|-----------|--------|----------|
| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
|**LogisticRegression** | 0.84     | 0.84      | 0.84   | 0.84     |
| Random Forest         | 0.83     | 0.83      | 0.82   | 0.83     |
| SVM                   | 0.83     | 0.83      | 0.83   | 0.83     |
|-----------------------|----------|-----------|--------|----------|


Penjelasan:
 - Ketiga model menunjukkan kinerja yang cukup baik, dengan F1-score di atas 0.83.
 - Logistic Regression sedikit unggul dalam keseimbangan precision dan recall, menjadikannya pilihan kuat untuk baseline yang ringan dan cepat.
 - Random Forest dan SVM memiliki performa serupa dan cocok untuk konteks ini karena mampu menangkap pola yang kompleks.
 - Karena ini adalah masalah deteksi dini kondisi mental, maka recall dan F1-score menjadi metrik paling krusial. Kesalahan dalam bentuk false negative (gagal mendeteksi siswa yang benar-benar depresi) dapat berdampak serius.
