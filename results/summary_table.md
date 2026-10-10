| Kode | Strategi | Trainable params | LR | Epoch konvergen | Epoch dijalankan | Waktu latih (s) | Val acc | Val macro F1 | Test acc | Test macro F1 | Train acc (clean) | Gap train−val |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | Baseline from scratch | 11,181,642 | all=0.001 | 14 | 19 | 156.5 | 0.4329 | 0.4176 | 0.4161 | 0.3955 | 0.4554 | 0.0225 |
| B2 | Feature extraction | 5,130 | head=0.001 | 8 | 13 | 92.2 | 0.8725 | 0.8707 | 0.8725 | 0.8706 | 0.9245 | 0.0520 |
| B3 | Progressive fine-tuning (layer4+fc) | 8,398,858 | last_block=0.0001; head=0.001 | 3 | 8 | 55.1 | 0.8826 | 0.8822 | 0.9228 | 0.9223 | 0.9698 | 0.0872 |
| B4a | B3 tanpa augmentasi | 8,398,858 | last_block=0.0001; head=0.001 | 5 | 10 | 31.9 | 0.8960 | 0.8953 | 0.9228 | 0.9213 | 1.0000 | 0.1040 |
| B4b | B3 dengan augmentasi | 8,398,858 | last_block=0.0001; head=0.001 | 3 | 8 | 55.2 | 0.8826 | 0.8822 | 0.9228 | 0.9223 | 0.9698 | 0.0872 |
| B5-EffB0 | efficientnet_b0 (finetune_last) | 3,168,550 | last_block=0.0001; head=0.001 | 3 | 8 | 59.4 | 0.9295 | 0.9294 | 0.9295 | 0.9293 | 0.9691 | 0.0395 |
| B5-MNv3S | mobilenet_v3_small (finetune_last) | 1,337,586 | last_block=0.0001; head=0.001 | 5 | 10 | 71.3 | 0.8591 | 0.8571 | 0.8960 | 0.8958 | 0.9554 | 0.0963 |