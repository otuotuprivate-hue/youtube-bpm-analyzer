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


# --- UI設計 ---
st.title("🎵 taetae-bpm-analyzer")
st.write(
    "音楽ファイルをアップロードすると、瞬時にBPMとKeyを高速解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
  st.audio(uploaded_file)

  if st.button("解析開始", type="primary"):
    with st.spinner("⚡ 高速解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          audio_path = os.path.join(temp_dir, uploaded_file.name)
          with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          # 爆速化のための読み込み（最初の45秒）
          y, sr = librosa.load(audio_path, sr=22050, duration=45)

          # BPM解析
          tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
          bpm = (
              float(tempo[0])
              if isinstance(tempo, np.ndarray)
              else float(tempo)
          )
          double_bpm = bpm * 2

          # Key解析
          key = estimate_key(y, sr)

          # 結果表示
          st.success("解析完了！")
          st.subheader(f"ファイル名: {uploaded_file.name}")

          col1, col2 = st.columns(2)
          with col1:
            st.metric(label="検出BPM", value=f"{bpm:.1f}")
          with col2:
            st.metric(label="Key（調）", value=key)

          # BPMの半テンポ・倍テンポに関する注釈ボックス
          with st.expander("💡 BPMの検知結果についての補足・確認"):
            st.write(
                f"- **倍テンポ（参考）**: `{double_bpm:.1f}`"
                f"（もしテンポが半分で検知されている場合はこちらが実際のBPMの可能性があります）"
            )
            st.caption(
                "楽曲の構成やビートの刻み方（4つ打ちかハーフタイムかなど）によって、"
                "解析エンジンが実際の半分または倍の値を返すことがあります。"
            )

        except Exception as e:
          st.error(f"解析エラーが発生しました:\n`{e}`")
