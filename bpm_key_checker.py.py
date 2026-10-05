import os
import re
import tempfile
import librosa
import numpy as np
import streamlit as st
import yt_dlp

# --- ページ設定 ---
st.set_page_config(
    page_title="YouTube BPM & Key Analyzer", page_icon="🎵", layout="centered"
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


def clean_youtube_url(url):
  # トラッキングパラメータや余分なクエリを完璧に除去
  url = re.sub(r"([&?]si=[^&]+)", "", url)
  url = re.sub(r"([&?]feature=[^&]+)", "", url)
  return url.strip()


# --- UI設計 ---
st.title("🎵 YouTube BPM & Key Analyzer")
st.write(
    "YouTube動画のURLを入力するだけで、BPMとKey（調）を自動解析します（極限突破モード）。"
)

url_input = st.text_input(
    "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("解析開始", type="primary"):
  if not url_input:
    st.warning("YouTubeのURLを入力してください。")
  else:
    target_url = clean_youtube_url(url_input)
    with st.spinner("🚀 極限突破バイパスで音声を解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        cookie_path = None
        try:
          # SecretsからCookieを一時ファイルへ書き出し
          if (
              "youtube" in st.secrets
              and "cookies" in st.secrets["youtube"]
          ):
            cookie_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w", encoding="utf-8"
            )
            cookie_file.write(st.secrets["youtube"]["cookies"])
            cookie_file.close()
            cookie_path = cookie_file.name

          # --- 最強のバイパスオプション群 ---
          ydl_opts = {
              "format": "bestaudio/best",
              "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
              "extractor_args": {
                  "youtube": {
                      # ブロックされにくい android_vr と ios クライアントを強制網羅
                      "player_client": ["android_vr", "ios", "web"],
                  }
              },
              "geo_bypass": True,
              "nocheckcertificate": True,
              "ignoreerrors": False,
              "postprocessors": [{
                  "key": "FFmpegExtractAudio",
                  "preferredcodec": "wav",
                  "preferredquality": "192",
              }],
              "quiet": True,
              "no_warnings": True,
          }

          if cookie_path and os.path.exists(cookie_path):
            ydl_opts["cookiefile"] = cookie_path

          with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(target_url, download=True)
            file_id = info["id"]
            title = info.get("title", "Unknown Title")
            wav_path = os.path.join(temp_dir, f"{file_id}.wav")

          # 万が一ファイル名が異なる場合のフォールバック
          if not os.path.exists(wav_path):
            files = os.listdir(temp_dir)
            wav_files = [os.path.join(temp_dir, f) for f in files if f.endswith(".wav")]
            if wav_files:
              wav_path = wav_files[0]
            else:
              raise FileNotFoundError(
                  "音声ファイルの抽出・変換に失敗しました。"
              )

          y, sr = librosa.load(wav_path, sr=22050, duration=90)

          # BPM解析
          tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
          bpm = (
              float(tempo[0])
              if isinstance(tempo, np.ndarray)
              else float(tempo)
          )

          # Key解析
          key = estimate_key(y, sr)

          # 結果表示
          st.success("解析が完了しました！")
          st.subheader(f"曲名: {title}")

          col1, col2 = st.columns(2)
          with col1:
            st.metric(label="BPM（テンポ）", value=f"{bpm:.1f}")
          with col2:
            st.metric(label="Key（調）", value=key)

        except Exception as e:
          st.error(f"エラーが発生しました:\n`{e}`")
          st.info(
              "💡 もしまだ弾かれる場合は、Streamlitダッシュボードのメニューから「Clear cache and reboot」を実行してから再度お試しください。"
          )
        finally:
          if cookie_path and os.path.exists(cookie_path):
            try:
              os.remove(cookie_path)
            except:
              pass
