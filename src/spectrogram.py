# spectrogram_analysis_v3.py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

DATA_PATH = Path("/data/raw")
OUTPUT_DIR = Path("/src/results/spectrogram_output")
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_spectrogram(wav_path, title, machine, section, domain, param, sr=16000):
    y, _ = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, hop_length=512, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{machine.upper()} | {title} | Section {section} | {domain} ({param})", fontsize=12)
    plt.tight_layout()
    
    save_name = f"{machine}_sec{section}_{domain}_{param}.png"
    plt.savefig(OUTPUT_DIR / save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_name}")

    # Phân tích định lượng
    mean_power = np.mean(S)
    peak_freq = librosa.mel_frequencies(n_mels=128)[np.argmax(np.mean(S, axis=1))]
    snr_est = np.max(S_dB) - np.percentile(S_dB, 10)

    print(f"  • Mean Power: {mean_power:.4f}")
    print(f"  • Peak Freq: {peak_freq:.1f} Hz")
    print(f"  • Est. SNR: {snr_est:.2f} dB\n")
    
    return snr_est

def compare_machine_section(machine, section):
    print(f"\n{'='*80}")
    print(f"{machine.upper()} - SECTION {section} - DOMAIN SHIFT")
    print(f"{'='*80}")
    
    attr = pd.read_csv(DATA_PATH / machine / f"attributes_{section}.csv")
    
    # Source normal
    src_row = attr[attr['file_name'].str.contains('source_train_normal')].iloc[0]
    src_path = DATA_PATH / src_row['file_name']
    src_param = src_row['d1v']
    
    # Target normal
    tgt_row = attr[attr['file_name'].str.contains('target_train_normal')].iloc[0]
    tgt_path = DATA_PATH / tgt_row['file_name']
    tgt_param = tgt_row['d1v']
    
    src_snr = analyze_spectrogram(src_path, "Source", machine, section, "source", src_param)
    tgt_snr = analyze_spectrogram(tgt_path, "Target", machine, section, "target", tgt_param)
    
    gap = src_snr - tgt_snr
    print(f"SNR GAP: {gap:+.2f} dB")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    for machine in ['fan', 'valve']:
        for sec in ['00', '01', '02']:
            compare_machine_section(machine, sec)
    
    print(f"HOÀN THÀNH! Tất cả ảnh lưu tại: {OUTPUT_DIR.resolve()}")