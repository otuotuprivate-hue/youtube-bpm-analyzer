import os
import tempfile
import librosa
import numpy as np
import streamlit as st

# --- ページ設定 ---
st.set_page_config(
    page_title="taetae-bpm-analyzer", page_icon="🎵", layout="centered"
)

# --- キー判定アルゴリズム（Krumhansl-Schmuckler プロファイル） ---
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
    """楽曲全体のコード感（クロマ特徴量）からKeyを高精度に判定する"""
    # CQTベースのクロマ特徴量を抽出
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
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


def estimate_overall_bpm(y, sr):
    """ドラム・リズム成分を分離して全体の正確なBPMを算出する"""
    # 楽器音をハーモニック（メロディ）とパーカッシブ（リズム）に分離
    _, y_percussive = librosa.effects.hpss(y, margin=2.0)

    # ビートトラッキングを実行
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # テンポの形式をスカラー値に変換
    if isinstance(tempo, np.ndarray):
        bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        bpm = float(tempo)

    # アイドルソング等でありがちな半テンポ・倍テンポの補正（75〜185の範囲に収める）
    while bpm < 75:
        bpm *= 2
    while bpm > 185:
        bpm /= 2

    return round(bpm, 1)


# --- UI設計 ---
st.title("🎵 taetae-bpm-analyzer")
st.write(
    "音楽ファイルをアップロードすると、曲全体の正確なBPMとKeyをシンプルに解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("解析開始", type="primary"):
        with st.spinner("🎧 曲全体を高精度解析中..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    audio_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 音声全体をロード
                    y, sr = librosa.load(audio_path, sr=22050)

                    # 全体のBPMとKeyを算出
                    bpm = estimate_overall_bpm(y, sr)
                    key = estimate_key(y, sr)
                    double_bpm = round(bpm * 2, 1)
                    half_bpm = round(bpm / 2, 1)

                    # 結果表示
                    st.success("解析完了！")
                    st.subheader(f"ファイル名: {uploaded_file.name}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="検出 BPM", value=f"{bpm:.1f}")
                    with col2:
                        st.metric(label="検出 Key", value=key)

                    # 参考用のテンポ候補（倍・半分）を表示
                    with st.expander("💡 テンポ（BPM）の微調整・確認用"):
                        st.write(
                            f"- **通常候補**: `{bpm:.1f}`\n"
                            f"- **倍テンポ候補**: `{double_bpm:.1f}`\n"
                            f"- **半テンポ候補**: `{half_bpm:.1f}`"
                        )
                        st.caption(
                            "楽曲のビートの刻み方（4つ打ち、倍テンポ感など）によって"
                            "実際のノリと数値が乖離する場合は、こちらの候補も参考にしてください。"
                        )

                except Exception as e:
                    st.error(f"解析エラーが発生しました:\n`{e}`")
