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
    """3連符や裏拍の誤認（1.5倍ズレ等）を対策した高精度BPM算出"""
    _, y_percussive = librosa.effects.hpss(y, margin=3.0)

    # オンセット強度を計算（細かい裏拍のノイズを拾いにくくする）
    onset_env = librosa.onset.onset_strength(
        y=y_percussive, sr=sr, hop_length=512, aggregate=np.median
    )

    # テンポ候補を複数取得するため、厳しめのスタート
    tempos = librosa.feature.tempo(
        onset_envelope=onset_env, sr=sr, aggregate=None
    )

    if len(tempos) > 0 and isinstance(tempos, np.ndarray):
        bpm = float(np.median(tempos))
    else:
        tempo_fallback, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)
        bpm = (
            float(tempo_fallback[0])
            if isinstance(tempo_fallback, np.ndarray)
            else float(tempo_fallback)
        )

    # --- 3連符・裏拍の誤認（約1.5倍ズレ）の自動補正ロジック ---
    # 110〜125あたりの数値で検出された場合、それは「170〜185の曲の3連符/裏拍勘違い」の可能性が非常に高い
    if 112 <= bpm <= 122:
        # 1.5倍にして本来のテンポ（約168〜183）に補正を試みる
        corrected = bpm * 1.5
        if 165 <= corrected <= 185:
            bpm = corrected

    # 一般的なレンジ（75〜185）に収まるようオクターブ調整
    while bpm < 75:
        bpm *= 2
    while bpm > 185:
        bpm /= 2

    return round(bpm, 1)


# --- UI設計 ---
st.title("🎵 taetae-bpm-analyzer")
st.write(
    "音楽ファイルをアップロードすると、3連符や裏拍の誤認を対策した高精度なBPMとKeyを解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
  st.audio(uploaded_file)

  if st.button("解析開始", type="primary"):
    with st.spinner("🎧 3連符・裏拍補正アルゴリズムで解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          audio_path = os.path.join(temp_dir, uploaded_file.name)
          with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          y, sr = librosa.load(audio_path, sr=22050)

          bpm = estimate_overall_bpm(y, sr)
          key = estimate_key(y, sr)
          double_bpm = round(bpm * 2, 1)
          half_bpm = round(bpm / 2, 1)

          st.success("解析完了！")
          st.subheader(f"ファイル名: {uploaded_file.name}")

          col1, col2 = st.columns(2)
          with col1:
            st.metric(label="検出 BPM", value=f"{bpm:.1f}")
          with col2:
            st.metric(label="検出 Key", value=key)

          with st.expander("💡 テンポ（BPM）の微調整・確認用"):
            st.write(
                f"- **通常候補**: `{bpm:.1f}`\n"
                f"- **1.5倍補正候補（3連符・裏拍対策）**: `{bpm * 1.5:.1f}`\n"
                f"- **倍テンポ候補**: `{double_bpm:.1f}`\n"
                f"- **半テンポ候補**: `{half_bpm:.1f}`"
            )
            st.caption(
                "3連符や裏拍を主軸とする楽曲で発生する特有のズレを自動補正するように調整しています。"
            )

        except Exception as e:
          st.error(f"解析エラーが発生しました:\n`{e}`")
                        )

                except Exception as e:
                    st.error(f"解析エラーが発生しました:\n`{e}`")
