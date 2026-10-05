import os
import tempfile
import librosa
import numpy as np
import streamlit as st

# --- ページ設定 ---
st.set_page_config(
    page_title="taetae-bpm-analyzer", page_icon="🎵", layout="centered"
)

# --- キー判定アルゴリズム ---
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.54, 3.98, 2.69, 3.34]
)
NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]


def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    if np.sum(chroma_sum) > 0:
        chroma_sum = chroma_sum / np.sum(chroma_sum)

    best_score = -1
    best_key = ""

    for i in range(12):
        maj_prof = np.roll(MAJOR_PROFILE, i)
        min_prof = np.roll(MINOR_PROFILE, i)

        corr_maj = np.corrcoef(chroma_sum, maj_prof)[0, 1]
        corr_min = np.corrcoef(chroma_sum, min_prof)[0, 1]

        if corr_maj > best_score:
            best_score = corr_maj
            best_key = f"{NOTE_NAMES[i]} Major ({NOTE_NAMES[i]})"

        if corr_min > best_score:
            best_score = corr_min
            best_key = f"{NOTE_NAMES[i]} Minor ({NOTE_NAMES[i]}m)"

    return best_key


def estimate_bpm_from_drums(y, sr):
    """ドラムの打点（アタック音）の間隔を直接解析してBPMを算出する"""
    _, y_percussive = librosa.effects.hpss(y, margin=3.0)

    onset_env = librosa.onset.onset_strength(
        y=y_percussive, sr=sr, hop_length=256, aggregate=np.sum
    )

    peak_frames = librosa.util.peak_pick(
        onset_env,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.2,
        wait=10,
    )
    peak_times = librosa.frames_to_time(peak_frames, sr=sr, hop_length=256)

    if len(peak_times) < 5:
        tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
        return float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)

    intervals = np.diff(peak_times)
    intervals = intervals[(intervals > 0.2) & (intervals < 1.5)]

    if len(intervals) == 0:
        return 120.0

    median_interval = np.median(intervals)
    calculated_bpm = 60.0 / median_interval

    while calculated_bpm < 75:
        calculated_bpm *= 2
    while calculated_bpm > 185:
        calculated_bpm /= 2

    return float(calculated_bpm)


# --- UI設計 ---
st.title("🎵 taetae-bpm-analyzer")
st.write(
    "【ドラム打点直読モード】楽曲のドラム・リズム隊の打撃音をダイレクトに解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("解析開始", type="primary"):
        with st.spinner("🥁 ドラムの打点（ビート）を解析中..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    audio_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    y, sr = librosa.load(audio_path, sr=22050)
                    total_duration = librosa.get_duration(y=y, sr=sr)

                    overall_bpm = estimate_bpm_from_drums(y, sr)
                    overall_key = estimate_key(y, sr)

                    st.success("解析完了！")
                    st.subheader(f"ファイル名: {uploaded_file.name}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="検出BPM（ドラム直読）", value=f"{overall_bpm:.1f}")
                    with col2:
                        st.metric(label="Key（調）", value=overall_key)

                    st.divider()
                    st.markdown("### 🎼 セクション別ドラム解析")

                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    bound_frames = librosa.segment.agglomerative(
                        mfcc, k=min(6, max(3, int(total_duration / 25)))
                    )
                    bound_times = librosa.frames_to_time(bound_frames, sr=sr)
                    bound_times = np.unique(
                        np.concatenate(([0.0], bound_times, [total_duration]))
                    )
                    bound_times.sort()

                    section_names = [
                        "イントロ",
                        "Aメロ",
                        "Bメロ",
                        "サビ",
                        "Cメロ / 間奏",
                        "大サビ / アウトロ",
                    ]

                    for i in range(len(bound_times) - 1):
                        start_time = bound_times[i]
                        end_time = bound_times[i + 1]

                        if (end_time - start_time) < 3.0:
                            continue

                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)
                        chunk_y = y[start_sample:end_sample]

                        chunk_bpm = estimate_bpm_from_drums(chunk_y, sr)
                        chunk_key = estimate_key(chunk_y, sr)

                        label = (
                            section_names[i]
                            if i < len(section_names)
                            else f"セクション {i+1}"
                        )

                        with st.container():
                            st.markdown(
                                f"**📍 {label}** (`{start_time:.1f}秒 〜 {end_time:.1f}秒`)"
                            )
                            sc_col1, sc_col2 = st.columns(2)
                            with sc_col1:
                                st.metric(label="BPM", value=f"{chunk_bpm:.1f}")
                            with sc_col2:
                                st.metric(label="Key", value=chunk_key)
                            st.markdown("---")

                except Exception as e:
                    st.error(f"解析エラーが発生しました:\n`{e}`")
