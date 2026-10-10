# Tugas 05 — Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Studi komparatif *training from scratch*, *feature extraction*, dan *progressive fine-tuning* pada ResNet-18 untuk klasifikasi 10 ras hewan peliharaan (subset Oxford-IIIT Pet), ditambah uji kontrol *data augmentation* dan perbandingan arsitektur efisien (EfficientNet-B0, MobileNetV3-Small).

## Struktur repositori

```
.
├── Tugas05_TransferLearning.ipynb   # notebook utama (B1–B5, C1, C3, C5)
├── laporan.md / laporan.pdf         # laporan analisis 3–5 halaman
├── results/                         # log mentah hasil eksekusi (dihasilkan notebook)
│   ├── config.json                  # konfigurasi + versi lingkungan
│   ├── split.csv                    # pembagian train/val/test per citra (bukti tanpa leakage)
│   ├── epoch_log.csv                # loss & akurasi per epoch untuk setiap kondisi
│   ├── summary.csv / summary.json   # metrik akhir per kondisi
│   ├── summary_table.md             # tabel konsolidasi siap tempel ke laporan
│   ├── test_predictions.csv         # prediksi test set setiap kondisi
│   ├── misclassified.csv            # sampel salah prediksi model terbaik
│   ├── layer4_weight_drift_B3.csv   # pergeseran bobot layer4 setelah fine-tuning
│   ├── runs/                        # log per kondisi (untuk RESUME)
│   └── figures/                     # seluruh grafik
├── requirements.txt
└── README.md
```

## Cara menjalankan

1. Buka notebook di Google Colab, pilih `Runtime → Change runtime type → T4 GPU`.
2. Opsional: set `SMOKE_TEST = True` dan jalankan semua sel untuk memastikan pipeline berjalan (± 2–3 menit, data sintetis, hasil disimpan terpisah di `results_smoke/`).
3. Set `SMOKE_TEST = False`, lalu `Runtime → Run all`. Dataset diunduh otomatis melalui `torchvision.datasets.OxfordIIITPet`.
4. Jika sesi Colab terputus, jalankan ulang semua sel: dengan `RESUME = True`, kondisi yang sudah selesai dimuat dari `results/runs/` dan `results/models/`.
5. Sel terakhir membuat `results_export.zip` (log + gambar, tanpa checkpoint model).

## Protokol eksperimen

| Item | Nilai |
|---|---|
| Kelas | Abyssinian, Bengal, Birman, Ragdoll, Siamese, American Pit Bull Terrier, Staffordshire Bull Terrier, Beagle, Basset Hound, Samoyed |
| Split | stratified 70 / 15 / 15 (`random_state=42`) dari gabungan split `trainval` + `test` torchvision |
| Input | 224 × 224, normalisasi ImageNet |
| Optimizer | AdamW, weight decay 1e-2, batch size 32 |
| Loss | CrossEntropyLoss |
| Early stopping | monitor val loss, patience 5, bobot terbaik dikembalikan |
| Seed | 42 (Python, NumPy, PyTorch, worker DataLoader; cuDNN deterministic) |
| Pemilihan model | berdasarkan Macro F1 **validasi**; test set hanya untuk pelaporan |

| Kode | Kondisi | Dilatih | LR |
|---|---|---|---|
| B1 | From scratch | seluruh jaringan | 1e-3 |
| B2 | Feature extraction | `fc` | 1e-3 |
| B3 | Progressive fine-tuning | `layer4` + `fc` | 1e-4 / 1e-3 |
| B4a / B4b | Konfigurasi B3 tanpa / dengan augmentasi | `layer4` + `fc` | 1e-4 / 1e-3 |
| B5 | Strategi terbaik B2/B3 pada EfficientNet-B0 dan MobileNetV3-Small | blok terakhir + head | 1e-4 / 1e-3 |

