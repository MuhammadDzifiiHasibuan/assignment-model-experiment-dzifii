# AI Model Experiment & Evaluation — Sentiment Analysis Ulasan Pelanggan

Membandingkan **model klasik Machine Learning (Scikit-learn)** dengan **LLM API (zero-shot prompting)** untuk mengklasifikasikan sentimen ulasan pelanggan e-commerce (positif/negatif), lalu menyusun rekomendasi technical approach berdasarkan hasil eksperimen.

**Ringkasan hasil:** pada evaluasi yang adil (leak-free), LLM (Groq `openai/gpt-oss-120b`) mencapai **100% accuracy / F1 1,000**, sedangkan model klasik TF-IDF + Logistic Regression hanya **64,5% accuracy / F1 0,636**. Rekomendasi: kembangkan pendekatan **LLM API**, dengan validasi tambahan pada ulasan nyata sebelum produksi.

## Struktur Project

```
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv     # dataset asli (tidak diubah)
├── notebook/
│   └── experiment_notebook.ipynb          # eksperimen kedua pendekatan + evaluasi + analisis (sudah berisi output)
├── results/
│   ├── llm_predictions_cache_groq_openai_gpt_oss_120b.csv   # prediksi mentah LLM per kalimat unik
│   └── model_comparison.csv               # tabel ringkasan metrik (dibuat otomatis oleh notebook)
├── documentation/
│   └── model_comparison_summary.png       # grafik perbandingan
├── .env.example
├── README.md
└── requirements.txt
```

## 1. Problem Statement

- **Objective:** membangun fitur otomatis yang mengklasifikasikan sentimen ulasan pelanggan pada halaman produk, dan memilih pendekatan teknis yang paling layak berdasarkan data.
- **Target/label:** `sentiment` ∈ {`positif`, `negatif`} (klasifikasi biner). Input: `review_text` (Bahasa Indonesia).
- **Batasan & asumsi:**
  - Label dataset dianggap ground truth; tidak ada kelas netral/campuran.
  - Dataset asli tidak dimodifikasi; kedua pendekatan dievaluasi pada data/test set yang sama.
  - Deep Learning dari nol di luar cakupan.

## 2. Dataset

`data/customer_reviews_sentiment.csv` — 200 baris, kolom `review_id`, `product_name`, `review_text`, `sentiment`.

| Label | Jumlah |
|---|---|
| positif | 110 |
| negatif | 90 |

> ⚠️ **Temuan penting:** dataset hanya berisi **40 kalimat unik** yang diulang rata-rata 5× (dengan nama produk berbeda); label tiap kalimat konsisten. Pada split acak 80/20 standar, **97,5% baris test memiliki kalimat yang sama persis dengan data train** (data leakage). Karena itu evaluasi dilakukan dalam dua skema (lihat bawah).

## 3. Ringkasan Eksperimen

**Skema evaluasi**

| Skema | Cara split | Fungsi |
|---|---|---|
| **A — Split standar** | `train_test_split` 80/20, stratified, `random_state=42` (160 train / 40 test) — sesuai starter notebook | Memenuhi brief; **mengandung leakage** |
| **B — Leak-free** | `StratifiedGroupKFold` 5-fold, grup = `review_text`; seluruh 200 baris diprediksi out-of-fold | **Acuan utama keputusan** — mengukur generalisasi ke kalimat baru |

**Pendekatan 1 — Model klasik:** TF-IDF (`TfidfVectorizer` default, vocabulary 145 kata pada skema A) + `LogisticRegression` (default). Baseline yang sederhana, cepat, dan mudah direproduksi.

**Pendekatan 2 — LLM API, zero-shot:** prompt berisi peran, definisi label, format output ketat (satu kata), dan petunjuk negasi (contoh negasi sengaja bukan kalimat dari dataset). Zero-shot dipilih karena task sederhana, sesuai skenario "tanpa training", dan menghindari leakage contoh few-shot pada dataset yang repetitif.

| Parameter | Nilai | Alasan |
|---|---|---|
| Provider / model | `groq` / `openai/gpt-oss-120b` | Free tier tanpa kartu kredit dan cepat. Provider dipindah dari Gemini karena kuota free tier Gemini cepat habis sebelum eksperimen selesai. |
| `temperature` | `0.0` (diterima API, tercatat `use_temperature=True`) | Klasifikasi butuh output deterministik & konsisten. |

Karena hanya ada 40 kalimat unik, prediksi LLM di-cache per kalimat unik → **40 panggilan API** cukup untuk seluruh evaluasi (skema A dan B). Seluruh 40 panggilan berhasil; output dinormalisasi dan **tidak ada output `invalid`**.

> **Catatan provider:** brief menyebut Gemini AI API. Eksperimen LLM dijalankan dengan Groq karena kendala kuota. Kode mendukung Gemini lewat `LLM_PROVIDER=gemini` di `.env` bila ingin mengulang dengan Gemini.

## 4. Hasil Evaluasi dan Perbandingan

Metrik = macro-average (kelas `negatif` & `positif` diperlakukan setara).

| Skema | Pendekatan | n | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| A (ada leakage) | Klasik (TF-IDF + LogReg) | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| A (ada leakage) | LLM groq/openai/gpt-oss-120b (zero-shot) | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **B (leak-free)** | Klasik (TF-IDF + LogReg) | 200 | 0.6450 | 0.6407 | 0.6359 | 0.6363 |
| **B (leak-free)** | **LLM groq/openai/gpt-oss-120b (zero-shot)** | 200 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

![Model comparison](documentation/model_comparison_summary.png)

**Confusion matrix (urutan: negatif, positif; baris = label asli, kolom = prediksi)**

| | Skema A (n=40) | Skema B (n=200) |
|---|---|---|
| Klasik | `[[18, 0], [0, 22]]` | `[[49, 41], [30, 80]]` |
| LLM | `[[18, 0], [0, 22]]` | `[[90, 0], [0, 110]]` |

**Interpretasi**
- **Skema A (seri 100%) tidak informatif:** model klasik cukup menghafal kalimat yang sudah dilihat saat training; LLM tidak dilatih pada dataset ini sehingga tidak terpengaruh.
- **Model klasik, skema B (64,5%):** hanya ±9,5 poin di atas tebakan kelas mayoritas (55%). Recall `negatif` 54,4% → hampir separuh keluhan pelanggan terlewat; precision `negatif` 62,0% → banyak ulasan bagus salah dicap negatif. Seluruh 71 kesalahan berasal dari 13 kalimat unik, terutama yang memuat **negasi** (mis. *"tidak mengecewakan"* → positif tetapi diprediksi negatif, salah 12×; *"tidak ada cacat sama sekali"*, salah 9×). TF-IDF bag-of-words tidak memahami urutan/makna kata dan kosakatanya sangat kecil.
- **LLM, skema B (100%):** 200 dari 200 benar; tidak ada false positive/negative. Seluruh 13 kalimat unik yang salah di model klasik diprediksi benar → LLM menangkap makna dan negasi. Skor identik di skema A dan B, sesuai harapan karena tidak ada training yang bisa "bocor".
- ⚠️ **Kehati-hatian:** 200 baris hanyalah pengulangan dari 40 kalimat unik (n efektif = 40) yang bersih dan polaritasnya jelas. Skor 100% tidak menjamin hasil sama pada ulasan nyata (sarkasme, slang, campur bahasa, netral/campuran).

**Kecepatan & biaya (terukur pada run ini)**

| Pendekatan | Setup/training | Waktu inferensi / ulasan | Biaya per prediksi |
|---|---|---|---|
| Klasik (TF-IDF + LogReg) | ±20 ms (sekali) | ±0,018 ms (lokal) | ≈ 0 |
| LLM groq/gpt-oss-120b | 0 (hanya desain prompt) | ±571 ms (rata-rata via jaringan) | Berbayar per token; input rata-rata ±185 token/panggilan, output termasuk reasoning token |

Estimasi biaya dalam uang belum dihitung karena harga per token belum diisi di notebook (Section 6.4).

## 5. Analisis Trade-off dan Limitation

| Aspek | Model klasik | LLM API |
|---|---|---|
| Performa (skema B) | Accuracy 64,5%, F1 0,636 | Accuracy 100%, F1 1,000 |
| Effort implementasi | Perlu data berlabel, training, evaluasi | Tanpa training; cukup desain prompt & parsing output |
| Kebutuhan data | Banyak & beragam (di sini hanya 40 kalimat unik → rapuh) | Praktis nol data berlabel |
| Kecepatan | ±0,018 ms/ulasan, lokal | ±571 ms/panggilan, ±30.000× lebih lambat, kena rate limit |
| Biaya | Sekali di muka; ≈ 0 per prediksi | Bayar per token di setiap prediksi; naik linear dengan volume |
| Kontrol & stabilitas | Deterministik, bisa di-host sendiri | Versi model bisa berubah/dipensiunkan; ketergantungan pihak ketiga |
| Privasi | Data tetap di infrastruktur sendiri | Teks ulasan dikirim ke pihak ketiga |

**Limitation**
- *Klasik (diamati):* leakage membuat skema A semu; tidak memahami negasi/konteks; kata di luar vocabulary diabaikan.
- *LLM:* **tidak ada kesalahan prediksi yang teramati** (0 dari 200), jadi keterbatasannya adalah risiko, bukan temuan: sarkasme, slang, campur bahasa, dan ulasan netral belum teruji. Kendala nyata yang diamati: **kuota free tier Gemini cepat habis**; selain itu latensi ±571 ms, biaya per token (reasoning token membuat jawaban satu kata tetap memakan banyak token), dan risiko deprecation model (Gemini 2.0 Flash dimatikan 1 Juni 2026).
- *Eksperimen:* dataset kecil dan template-like (n efektif 40); hanya dua kelas; satu prompt, satu model, satu run per pendekatan; provider LLM berbeda dari yang disebut di brief.

## 6. Rekomendasi Technical Approach

**Rekomendasi: kembangkan lebih lanjut pendekatan LLM API (zero-shot prompting), dengan validasi tambahan pada ulasan nyata sebelum rilis produksi.**

Alasan (didukung hasil eksperimen):
1. **Performa jauh lebih baik pada evaluasi yang adil:** 100% vs 64,5% accuracy (F1 1,000 vs 0,636) pada skema leak-free; LLM benar pada seluruh kalimat yang salah di model klasik, termasuk yang bernegasi.
2. **Model klasik tidak generalisasi pada data ini:** skor 100% di skema A hanyalah efek leakage; pada ulasan baru ia hanya sedikit di atas menebak kelas mayoritas (55%).
3. **Tidak butuh data berlabel besar:** hanya ada 40 kalimat unik; LLM langsung bisa dipakai.
4. **Time-to-market cepat dan hasil konsisten:** tanpa training; `temperature=0` memberi 0 output `invalid`.
5. **Biaya dan latensi dapat diterima di fase awal:** ±571 ms/panggilan cocok untuk pemrosesan asinkron/batch dan hasil dapat di-cache per ulasan.

Mitigasi & langkah lanjutan:
- **Validasi lebih keras:** buat test set ulasan nyata berlabel manusia (sarkasme, slang, campur bahasa, netral/campuran) dan ukur ulang — skor 100% di sini berasal dari 40 kalimat bersih.
- **Hitung biaya nyata:** isi harga per token di notebook lalu proyeksikan ke volume ulasan per bulan.
- Kunci versi model + pantau deprecation, rencanakan kuota/rate limit (batching, retry, cache), dan sediakan fallback saat API gagal.
- Ulangi dengan Gemini (sesuai brief) bila diperlukan untuk memastikan hasil setara.

Kapan berubah: jika volume sangat besar, latensi harus sangat rendah, atau data tidak boleh keluar infrastruktur → latih model klasik/fine-tuned dengan data jauh lebih besar dan beragam (label dibantu LLM lalu dikurasi manusia — pendekatan hybrid). Jika LLM ternyata jauh lebih buruk pada ulasan nyata, iterasi prompt (few-shot) dahulu.

## 7. Cara Menjalankan

```bash
# 1. (opsional) buat virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependency
pip install -r requirements.txt

# 3. siapkan provider LLM
cp .env.example .env               # lalu edit .env: pilih LLM_PROVIDER dan isi key-nya

# 4. jalankan notebook
jupyter notebook notebook/experiment_notebook.ipynb
```

Pilih **Run All**.

**Pilihan provider LLM** (atur `LLM_PROVIDER` di `.env`):

| Provider | Key | Catatan |
|---|---|---|
| `groq` (default; dipakai pada hasil di atas) | `GROQ_API_KEY` dari https://console.groq.com/keys | Free tier tanpa kartu kredit, cepat |
| `gemini` | `GEMINI_API_KEY` | Sesuai brief; free tier cepat habis |
| `openrouter` | `OPENROUTER_API_KEY` | Model `:free`, batas harian ketat |
| `ollama` | tanpa key | Model lokal: install Ollama, `ollama pull llama3.2:3b`; tanpa kuota |

Catatan:
- Prediksi LLM disimpan di `results/llm_predictions_cache_<provider>_<model>.csv`. Jika file ini ada, menjalankan ulang **tidak memanggil API lagi**; hapus file tersebut untuk memanggil ulang.
- Tanpa cache dan tanpa key yang sesuai, sel LLM dilewati dengan pesan jelas (tanpa error) dan hanya model klasik yang dievaluasi.
- Jika kuota harian habis di tengah jalan, notebook berhenti rapi; jalankan ulang sel 5.3 nanti (melanjutkan dari cache) atau ganti provider.
- Untuk mengganti model: set `LLM_MODEL` di `.env` (katalog model gratis sering berubah; cek dashboard provider).
- Untuk estimasi biaya, isi `PRICE_INPUT_PER_1M` dan `PRICE_OUTPUT_PER_1M` di Section 6.4 dengan harga resmi provider.
- **Jangan commit file `.env`** (sudah ada di `.gitignore`).
# assignment-model-experiment-dzifii
