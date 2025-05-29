# Laporan Proyek Machine Learning - Saftana Fitri

## Project Overview

Di era digital saat ini, ledakan informasi dan konten telah menghadirkan tantangan baru bagi pengguna dalam menemukan produk atau layanan yang relevan. Dalam industri hiburan, khususnya platform streaming film atau penyedia konten video, pengguna seringkali dihadapkan pada katalog yang sangat luas, berisi ribuan hingga jutaan judul film. Hal ini dapat menyebabkan *information overload*, di mana pengguna merasa kewalahan dan kesulitan untuk memutuskan film mana yang akan ditonton.

Masalah ini akan berdampak pada tingkat keterlibatan (engagement) dan retensi pengguna terhadap platform. Jika pengguna kesulitan menemukan film yang sesuai dengan preferensi mereka, maka besar kemungkinan mereka akan berhenti menggunakan layanan tersebut. Untuk menjawab tantangan tersebut, dibutuhkan sebuah sistem rekomendasi yang mampu memberikan saran film yang relevan dan dipersonalisasi berdasarkan data historis, karakteristik konten, atau perilaku pengguna lain yang memiliki kesamaan selera.

Ricci et al. menyatakan bahwa penggunaan sistem rekomendasi yang dirancang secara efektif mampu meningkatkan click-through rate, interaksi pengguna, dan kepuasan layanan [1]. Sementara itu, Deldjoo et al. dalam survei mereka menekankan pentingnya pemilihan metode rekomendasi yang sesuai dengan konteks, seperti collaborative filtering dan content-based filtering, yang masing-masing memiliki kelebihan dan kekurangan tergantung pada struktur data dan metadata yang tersedia [2].

Dengan memanfaatkan metode collaborative filtering dan content-based filtering, sistem rekomendasi film ini diharapkan dapat menghasilkan top-N rekomendasi yang akurat, relevan, dan meningkatkan pengalaman pengguna dalam menjelajahi katalog film yang luas.

**Referensi**:

[1] F. Ricci, L. Rokach, dan B. Shapira, Recommender Systems Handbook, 3rd ed. Cham, Switzerland: Springer, 2022.

[2] Y. Deldjoo, T. Di Noia, dan B. Kille, “Content-based Video Recommendation Systems: A Survey,” ACM Trans. Multimedia Comput. Commun. Appl., vol. 16, no. 2, pp. 1–34, 2020, doi: 10.1145/3383184.

## Business Understanding

### Problem Statements

- Bagaimana cara memberikan rekomendasi film kepada pengguna berdasarkan kemiripan fitur atau konten film (seperti genre) dengan film-film lain yang ada dalam katalog?
- Bagaimana cara memberikan rekomendasi film kepada pengguna berdasarkan pola perilaku dan preferensi mereka (yang tercermin dalam rating yang diberikan) serta pola perilaku pengguna lain yang memiliki selera serupa?
- Bagaimana mengembangkan dan membandingkan dua pendekatan sistem rekomendasi yang berbeda (Content-Based Filtering dan Collaborative Filtering) untuk memberikan rekomendasi film?

### Goals

- Mengembangkan model sistem rekomendasi film menggunakan pendekatan Content-Based Filtering yang mampu menyarankan film berdasarkan kesamaan genre.
- Mengembangkan model sistem rekomendasi film menggunakan pendekatan Collaborative Filtering (dengan teknik faktorisasi matriks berbasis model embedding) yang mampu menyarankan film berdasarkan histori rating pengguna.
- Menghasilkan daftar top-N rekomendasi film untuk pengguna atau film tertentu dari kedua model yang dikembangkan.
- Mengevaluasi kinerja model Collaborative Filtering menggunakan metrik kuantitatif seperti Root Mean Squared Error (RMSE).

### Solution statements

1. Pendekatan Content-Based Filtering:
- Menggunakan fitur genre dari setiap film sebagai dasar konten. Genre film akan diekstrak dan diproses
- Menerapkan teknik TF-IDF (Term Frequency-Inverse Document Frequency) pada data genre untuk mengubah deskripsi tekstual genre menjadi representasi vektor numerik yang dapat diukur kesamaannya.
- Menggunakan metrik *Cosine Similarity* untuk menghitung derajat kesamaan antara vektor fitur genre dari semua pasangan film.
- Berdasarkan film input (misalnya, film yang sedang dilihat atau disukai pengguna), sistem akan merekomendasikan film-film lain yang memiliki skor kesamaan genre tertinggi.

2. Pendekatan Collaborative Filtering:
- Menggunakan data historis rating yang diberikan oleh pengguna terhadap film.
- Melakukan encoding pada user_id dan item_id (ID film) menjadi indeks integer yang sekuensial agar dapat digunakan sebagai input pada lapisan embedding.
- Membangun model neural network menggunakan TensorFlow/Keras yang memiliki lapisan Embedding terpisah untuk pengguna dan film, mekanisme interaksi antara embedding pengguna dan film, dan bias untuk pengguna dan film untuk menangkap variasi intrinsik.
- Model dilatih untuk memprediksi rating yang akan diberikan pengguna terhadap film. Fungsi loss (misalnya, Mean SquaredError) akan diminimalkan selama proses training.
- Untuk pengguna target, model akan memprediksi rating untuk film-film yang belum pernah ia tonton. Film dengan prediksi rating tertinggi akan direkomendasikan.

## Data Understanding

Dataset yang digunakan adalah MovieLens 100k yang dikumpulkan dan dikelola oleh [GroupLens](https://grouplens.org/datasets/movielens/100k/). Jumlah Data yang ada di dataset ini adalah 100.000 baris data rating dari 943 pengguna untuk 1682 film. Rating diberikan dalam skala integer 1 hingga 5, serta menyediakan informasi lainnya seperti demografis pengguna dan metadata film. Dataset dimuat ke dalam tiga DataFrame utama: `ratings_df`, `movies_df`, dan `users_df`.

1. **ratings_df**:
   - `user_id`: ID unik pengguna
   - `item_id`: ID unik film
   - `rating`: Nilai rating (1-5)
   - `timestamp`: Waktu rating diberikan

2. **movies_df** (metadata film):
   - `item_id`: ID unik film
   - `movie_title`: Judul film
   - `release_date`: Tanggal rilis
   - `video_release_date`: Tanggal rilis video (jika ada)
   - `IMDb_url`: URL IMDb
   - `genre flags`: 19 kolom biner (1 atau 0) yang mewakili genre

3. **users_df** (data pengguna):
   - `user_id`: ID unik pengguna
   - `age`: Umur pengguna
   - `gender`: Jenis kelamin pengguna
   - `occupation`: Pekerjaan pengguna
   - `zip_code`: Kode pos

### EDA (Exploratory Data Analysis)

- Distribusi Rating: Sebagian besar rating yang diberikan adalah 4 (34.174 rating) dan 3 (27.145 rating), menunjukkan kecenderungan pengguna memberikan rating di atas rata-rata. Rating 5 diberikan sebanyak 21.201 kali.

```
# Distribusi Rating:
# rating
# 1     6110
# 2    11370
# 3    27145
# 4    34174
# 5    21201
# Name: count, dtype: int64
```

- Jumlah Film per Gende: Genre Drama adalah yang paling banyak (725 film), diikuti oleh Comedy (505 film), Action (251 film), dan Thriller (251 film). Genre 'unknown' hanya ada pada 2 film.

- Demografi Pengguna: Terdapat 670 pengguna laki-laki (M) dan 273 pengguna perempuan (F), pekerjaan paling umum adalah 'student' (196 pengguna), diikuti oleh 'other' (105 pengguna) dan 'educator' (95 pengguna), dan Usia: rata-rata usia pengguna adalah sekitar 34 tahun, dengan standar deviasi 12.19 tahun. Usia termuda adalah 7 tahun dan tertua 73 tahun.

```
python
# users_df['age'].describe()
# count    943.000000
# mean      34.051962
# std       12.192740
# min        7.000000
# 25%       25.000000
# 50%       31.000000
# 75%       43.000000
# max       73.000000
# Name: age, dtype: float64
```

- Judul Film Unik: Terdapat 1682 `item_id` unik, namun hanya 1664 `movie_title` unik di `movies_df`. Ini mengindikasikan kemungkinan ada beberapa film dengan judul yang sama namun `item_id` berbeda, atau ada sedikit ketidakkonsistenan pada data judul.

## Data Preparation

Tahap persiapan data sangat krusial untuk memastikan data siap digunakan oleh model machine learning. Proses-proses yang dilakukan :

1.  Verifikasi Entitas Unik:
    Dilakukan pengecekan jumlah `user_id` unik dan `item_id` unik pada masing-masing DataFrame (`ratings_df`, `movies_df`, `users_df`) untuk memastikan konsistensi data dan memahami skala dataset.
    
    **Alasan**: Untuk memastikan konsistensi data dan memahami skala dataset. Ditemukan 943 pengguna unik dan 1682 film unik yang terlibat dalam data rating, yang sesuai dengan jumlah total pengguna dan film dalam dataset.

2.  Penggabungan DataFrame (Merging):
    `ratings_df` digabungkan dengan `movies_df` menggunakan `item_id` sebagai kunci. Hasilnya disimpan sebagai `df_ratings_with_movie_info`. ```python
    genre_columns = ['unknown', 'Action', ..., 'Western']
    columns_to_merge_movies = ['item_id', 'movie_title'] + genre_columns
    df_ratings_with_movie_info = pd.merge(ratings_df, movies_df[columns_to_merge_movies], on='item_id', how='left')
    ```
    df_ratings_with_movie_info` kemudian digabungkan dengan `users_df` menggunakan `user_id` sebagai kunci. Hasilnya disimpan sebagai `df_all_info`. 
    ```python
    df_all_info = pd.merge(df_ratings_with_movie_info,users_df, on='user_id', how='left')
    ```
    
    **Alasan**: Penggabungan ini penting untuk menciptakan satu dataset terpadu yang kaya akan fitur, yang dapat digunakan oleh kedua model rekomendasi. Model Content-Based akan menggunakan fitur film (genre, judul), sedangkan Collaborative Filtering akan menggunakan interaksi pengguna-film (rating) dan bisa juga diperkaya dengan fitur pengguna.

3.  Penanganan Missing Value:
    Setelah penggabungan, dilakukan pengecekan nilai null pada `df_all_info` menggunakan `df_all_info.isnull().sum()`.
    ```
    # Output dari df_all_info.isnull().sum()
    # user_id        0
    # item_id        0
    # rating         0
    # timestamp      0
    # movie_title    0
    # ... (semua kolom genre) ... 0
    # age            0
    # gender         0
    # occupation     0
    # zip_code       0
    # dtype: int64
    ```
    
    **Alasan**: Untuk memastikan tidak ada data yang hilang pada kolom-kolom yang akan digunakan. Jika ada, perlu strategi penanganan (misalnya, imputasi atau penghapusan).
    
    **Hasil**: Ditemukan bahwa `df_all_info` tidak memiliki missing value pada kolom-kolom hasil penggabungan yang relevan (seperti `user_id`, `item_id`, `rating`, `movie_title`, kolom genre, `age`, `gender`, `occupation`). Ini menunjukkan bahwa semua `item_id` di `ratings_df` ada di `movies_df`, dan semua `user_id` di `ratings_df` (yang sudah digabung dengan info film) ada di `users_df`.

4.  Konversi Tipe Data `timestamp`:
    Kolom `timestamp` (yang merupakan Unix timestamp integer) dikonversi menjadi objek `datetime` Pandas.
    
    ```python
    df_all_info['datetime'] = pd.to_datetime(df_all_info['timestamp'], unit='s')
    ```
    
    **Alasan**: Mengubah timestamp menjadi format datetime yang lebih mudah dibaca dan dapat digunakan untuk analisis atau *feature engineering* berbasis waktu.

5.  Persiapan Data untuk Content-Based Filtering:
    DataFrame `df_fixed_movies` dibuat dengan mengurutkan `df_all_info` berdasarkan `item_id`. Lalu, `movie_features_df` dibuat dengan mengambil kolom `item_id`, `movie_title`, dan kolom-kolom genre dari `df_fixed_movies`, lalu menghapus baris duplikat berdasarkan `item_id`. Sebuah fungsi `get_genre_strings` dibuat untuk mengkonversi 19 kolom genre biner menjadi satu kolom string `genre_string` di mana setiap genre aktif dipisahkan oleh karakter `|`. Terakhir, dataFrame `movies_prepared_df` dibuat dengan kolom `id` (sebelumnya `item_id`), `movie_title`, dan `genres` (kolom `genre_string`).

    **Alasan**: Untuk membuat dataset film yang ringkas dengan fitur genre dalam format teks yang siap diproses oleh `TfidfVectorizer` untuk model Content-Based Filtering.

6.  **Persiapan Data untuk Collaborative Filtering**:
    DataFrame `ratings_prepared_df` dibuat dengan memilih hanya kolom `user_id`, `item_id`, dan `rating` dari `df_all_info`. Lalu, **Encoding ID**: `user_id` dan `item_id` asli di-encode menjadi indeks integer yang dimulai dari 0. Ini dilakukan dengan membuat pemetaan (dictionary) dari ID asli ke ID encode, dan sebaliknya. Kolom baru `user_encoded` dan `movie_encoded` ditambahkan ke DataFrame `df` (salinan dari `ratings_prepared_df`).

    ```python
    user_ids_unique_list = df['user_id'].unique().tolist()
    user_to_user_encoded = {original_id: i for i, original_id in enumerate(user_ids_unique_list)}
    # ... (dan seterusnya untuk user_encoded_to_user, movie_to_movie_encoded, movie_encoded_to_movie)
    df['user_encoded'] = df['user_id'].map(user_to_user_encoded)
    df['movie_encoded'] = df['item_id'].map(movie_to_movie_encoded)
    ```
    Setelah itu, Kolom `rating` dikonversi ke `np.float32`. Nilai rating dinormalisasi ke rentang [0, 1] menggunakan min-max scaling: `y = df['rating'].apply(lambda r: (r - min_rating) / (max_rating - min_rating)).values`. Dataset `df` diacak, kemudian dibagi menjadi data training (80%) dan data validasi (20%) untuk fitur (`x`, yaitu pasangan `[user_encoded, movie_encoded]`) dan target (`y`, yaitu rating yang dinormalisasi).
    
    **Alasan**:
    * Encoding ID diperlukan agar ID pengguna dan film dapat digunakan sebagai input untuk lapisan `Embedding` dalam model neural network.
    * Normalisasi rating membantu dalam proses training model, terutama jika fungsi aktivasi output adalah sigmoid (yang menghasilkan nilai antara 0 dan 1).
    * Pembagian data menjadi set training dan validasi adalah praktik standar dalam machine learning untuk melatih model dan mengevaluasi kemampuannya dalam melakukan generalisasi pada data baru.

## Modeling
Dua model sistem rekomendasi dikembangkan: Content-Based Filtering dan Collaborative Filtering.

### 1. Model Content-Based Filtering
Model ini merekomendasikan film berdasarkan kemiripan genre.

-   **Proses Pemodelan**:
    1.  **Representasi Fitur Genre dengan TF-IDF**: Kolom `genres` (string genre yang dipisahkan `|`) dari `movies_prepared_df` digunakan. Karakter `|` diganti spasi. `TfidfVectorizer` dari `sklearn` digunakan untuk mengubah string genre menjadi matriks TF-IDF. Matriks ini memiliki dimensi (jumlah film x jumlah genre unik), di mana setiap entri adalah skor TF-IDF yang menunjukkan pentingnya suatu genre untuk film tersebut.
        ```python
        # data_model = movies_prepared_df.copy()
        # data_model['genres_for_tfidf'] = data_model['genres'].str.replace('|', ' ', regex=False)
        # tf = TfidfVectorizer()
        # tfidf_matrix = tf.fit_transform(data_model['genres_for_tfidf'])
        # Output shape: (1682, 21) (1682 film, 21 fitur genre unik)
        ```
    2.  **Perhitungan Kesamaan (Cosine Similarity)**: `cosine_similarity` dihitung pada `tfidf_matrix` untuk mendapatkan matriks kesamaan antar film (`cosine_sim_df`) berukuran (1682 x 1682).
        ```python
        # cosine_sim = cosine_similarity(tfidf_matrix)
        # cosine_sim_df = pd.DataFrame(cosine_sim, index=data_model['movie_title'], columns=data_model['movie_title'])
        ```
    3.  **Fungsi Rekomendasi**: Sebuah fungsi `movie_recommendations(movie_title, k=5)` dibuat. Fungsi ini mengambil judul film sebagai input, mencari film tersebut dalam `cosine_sim_df`, mengambil skor kesamaannya dengan semua film lain, mengurutkannya, dan mengembalikan `k` film teratas yang paling mirip (tidak termasuk film input itu sendiri).

-   **Hasil (Top-N Recommendation)**:
    Sebagai contoh, untuk film **'Toy Story (1995)'** (Genre: Animation|Children|Comedy), 5 rekomendasi teratas yang dihasilkan adalah:
    ```
    # Output dari movie_recommendations('Toy Story (1995)', k=5)
    #                                          movie_title                      genres    id
    # 0             Aladdin and the King of Thieves (1996)   Animation|Children|Comedy   422
    # 1                            Gumby: The Movie (1995)          Animation|Children  1470
    # 2  Land Before Time III: The Time of the Great Gi...          Animation|Children  1412
    # 3                            Oliver & Company (1988)          Animation|Children  1078
    # 4                     Sword in the Stone, The (1963)          Animation|Children   625
    ```

    **Kelebihan**:
    * Tidak memerlukan data dari pengguna lain (mengatasi *user cold-start* jika profil awal pengguna dapat dibuat).
    * Dapat merekomendasikan item baru atau kurang populer (*niche items*) selama fitur kontennya tersedia.
    * Rekomendasi bersifat transparan dan dapat dijelaskan berdasarkan kesamaan fitur (misalnya, "direkomendasikan karena sama-sama bergenre Drama").
    
    **Kekurangan**:
    * Kualitas sangat bergantung pada kualitas ekstraksi fitur konten.
    * Cenderung menghasilkan rekomendasi yang terlalu mirip (*filter bubble*), kurang memberikan kejutan (*serendipity*).
    * Sulit menangani *item cold-start* jika item baru belum memiliki deskripsi fitur yang memadai.

### 2. Model Collaborative Filtering
Model ini merekomendasikan film berdasarkan pola rating pengguna menggunakan pendekatan neural network dengan embedding.

-   **Arsitektur Model (`RecommenderNet`)**:
    * Model didefinisikan sebagai kelas `RecommenderNet` yang mewarisi `tf.keras.Model`.
    * **Input**: Pasangan `[user_encoded, movie_encoded]`.
    * **Lapisan Embedding**:
        * `user_embedding`: Mengubah `user_encoded` menjadi vektor embedding (ukuran `embedding_size`, misal 20).
        * `user_bias`: Vektor bias untuk setiap pengguna (ukuran 1).
        * `item_embedding` (untuk film): Mengubah `movie_encoded` menjadi vektor embedding.
        * `item_bias`: Vektor bias untuk setiap film.
        * Semua lapisan embedding menggunakan inisialisasi `he_normal` dan regularisasi L2 (`1e-6`).
    * **Interaksi**: Hasil interaksi (dot product) tersebut kemudian dijumlahkan dengan user_bias dan item_bias untuk memperhitungkan kecenderungan rating intrinsik dari masing-masing pengguna dan film. Sebuah lapisan Dropout(0.3) diterapkan setelah penjumlahan bias untuk regularisasi, yang membantu mencegah overfitting. Terakhir, output dilewatkan melalui lapisan Dense(1, activation=None) yang menghasilkan prediksi rating linear. Meskipun target y (rating) telah dinormalisasi ke rentang [0,1] pada tahap persiapan data, penggunaan output linear dengan loss function MeanSquaredError (MSE) adalah pendekatan yang valid untuk masalah regresi prediksi rating ini. Model akan belajar untuk memprediksi nilai dalam rentang ternormalisasi tersebut.

        ```python
        # class RecommenderNet(tf.keras.Model):
        #     def __init__(self, num_users, num_items, embedding_size, **kwargs):
        #         # ... (definisi embedding dan bias) ...
        #         self.dropout = layers.Dropout(0.3)
        #         self.dense = layers.Dense(1, activation=None)
        #
        #     def call(self, inputs, training=False):
        #         # ... (lookup embedding dan bias) ...
        #         dot_user_item = tf.reduce_sum(user_vector * item_vector, axis=1, keepdims=True)
        #         x = dot_user_item + user_bias + item_bias
        #         x = self.dropout(x, training=training)
        #         x = self.dense(x)
        #         return x
        ```

-   **Proses Training**:
    * Model diinisialisasi dengan `num_users` (943), `num_movies` (1682 hasil encoding dari film yang dirating), dan `embedding_size` (20).
    * Model dikompilasi dengan `loss=tf.keras.losses.MeanSquaredError()`, optimizer `Adam(learning_rate=0.0005)`, dan metrik `RootMeanSquaredError()`.
    * Callback `EarlyStopping` digunakan untuk memantau `val_loss` dengan `patience=4`.
    * Model dilatih selama `epochs=20` dengan `batch_size=64` menggunakan data `x_train` dan `y_train` (rating yang dinormalisasi 0-1), serta divalidasi dengan `x_val` dan `y_val`.

-   **Hasil (Top-N Recommendation)**:
    * Untuk seorang pengguna acak (misalnya, `user_id` asli 104), model memprediksi rating untuk film-film yang belum pernah ditonton oleh pengguna tersebut.
    * 10 film dengan prediksi rating tertinggi kemudian direkomendasikan.
    * **Contoh Film yang Pernah Dirating Tinggi oleh User 104**:
        ```
        # - Amistad (1997) (Rating: 5.0), Genres: Drama
        # - Swingers (1996) (Rating: 5.0), Genres: Comedy|Drama
        # - Star Wars (1977) (Rating: 5.0), Genres: Action|Adventure|Romance|Sci-Fi|War
        # - L.A. Confidential (1997) (Rating: 5.0), Genres: Crime|Film-Noir|Mystery|Thriller
        # - Return of the Jedi (1983) (Rating: 5.0), Genres: Action|Adventure|Romance|Sci-Fi|War
        ```
    * **Contoh Top 10 Rekomendasi Film untuk User 104**:
        ```
        # - World of Apu, The (Apur Sansar) (1959), Genres: Drama
        # - Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963), Genres: Sci-Fi|War
        # - Eat Drink Man Woman (1994), Genres: Comedy|Drama
        # - Bridge on the River Kwai, The (1957), Genres: Drama|War
        # - To Kill a Mockingbird (1962), Genres: Drama
        # - Eve's Bayou (1997), Genres: Drama
        # - Shawshank Redemption, The (1994), Genres: Drama
        # - Raise the Red Lantern (1991), Genres: Drama
        # - Notorious (1946), Genres: Film-Noir|Romance|Thriller
        # - Late Bloomers (1996), Genres: Comedy
        ```
    
    **Kelebihan**:
    * Tidak memerlukan fitur konten item secara eksplisit; model belajar fitur laten dari pola interaksi.
    * Mampu menemukan rekomendasi yang lebih beragam dan mengejutkan (*serendipitous*) karena menangkap kemiripan berdasarkan perilaku pengguna.
    * Umumnya memberikan performa yang baik jika data rating cukup banyak dan padat.
    
    **Kekurangan**:
    * Sangat rentan terhadap masalah *data sparsity* (data rating yang sedikit).
    * Mengalami kesulitan dengan *item cold-start* (item baru tanpa rating tidak bisa direkomendasikan).
    * Mengalami kesulitan dengan *user cold-start* (pengguna baru tanpa histori rating tidak bisa mendapatkan rekomendasi personal).
    * Rekomendasi seringkali kurang transparan (bersifat *black box*), sulit dijelaskan mengapa suatu item direkomendasikan.

## Evaluation

-   **Content-Based Filtering**:
    * Evaluasi untuk model ini dalam proyek ini bersifat **kualitatif**. Dengan melihat output rekomendasi untuk film 'Toy Story (1995)' (genre Animation, Children, Comedy), film-film yang direkomendasikan seperti 'Aladdin and the King of Thieves' dan 'Gumby: The Movie' juga memiliki genre Animation dan Children, yang menunjukkan relevansi berdasarkan konten.

-   **Collaborative Filtering**:
    * Metrik evaluasi utama yang digunakan adalah **Root Mean Squared Error (RMSE)** dan **Mean Squared Error (MSE)** sebagai *loss function*.
    * **Formula RMSE**:
        RMSE mengukur akar dari rata-rata kuadrat perbedaan antara nilai rating aktual ($y_i$) dan nilai rating yang diprediksi ($\hat{y}_i$) oleh model.
        $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
        Di mana $N$ adalah jumlah total rating yang dievaluasi. Nilai RMSE yang lebih rendah menunjukkan bahwa prediksi model lebih dekat dengan nilai aktual, sehingga model dianggap lebih baik. RMSE memberikan bobot lebih pada kesalahan yang besar karena adanya pengkuadratan.
    * **Cara Kerja Metrik**: Selama training, model mencoba meminimalkan MSE (dan secara implisit RMSE) pada data training. Kinerja pada data validasi (RMSE validasi dan loss validasi) dipantau untuk melihat seberapa baik model melakukan generalisasi pada data yang belum pernah dilihat. Jika RMSE validasi mulai meningkat sementara RMSE training terus menurun, itu adalah tanda overfitting.
    * **Hasil Proyek Berdasarkan Metrik (Collaborative Filtering)**:
        Berdasarkan plot training yang dihasilkan:
        * Model dihentikan oleh `EarlyStopping` pada epoch ke-9 (dari 20 epoch yang direncanakan), yang mengindikasikan bahwa `val_loss` tidak lagi membaik.
        * Pada epoch terakhir (epoch 9):
            * `loss` (MSE training): sekitar 0.0445
            * `root_mean_squared_error` (RMSE training): sekitar 0.2120
            * `val_loss` (MSE validasi): sekitar 0.0587
            * `val_root_mean_squared_error` (RMSE validasi): sekitar 0.2412

        **Interpretasi Hasil**:
        Nilai val_root_mean_squared_error terbaik yang dicapai adalah sekitar 0.2367. Mengingat target rating y dinormalisasi ke rentang [0, 1], nilai RMSE ini menunjukkan rata-rata kesalahan prediksi. Untuk menginterpretasikannya dalam skala rating asli (1-5, dengan rentang 4 poin), kita dapat mengalikannya: 0.2367×(5−1)=0.2367×4≈0.9468. Ini berarti, secara rata-rata, prediksi rating model memiliki kesalahan sekitar 0.95 poin pada skala rating 1-5. Ini merupakan performa yang cukup baik untuk model sistem rekomendasi, yang menunjukkan model mampu mempelajari pola preferensi pengguna dari data rating yang ada. 