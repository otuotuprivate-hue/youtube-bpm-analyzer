import os
import re
import tempfile
import librosa
import numpy as np
import requests
import streamlit as st

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
  url = re.sub(r"([&?]si=[^&]+)", "", url)
  url = re.sub(r"([&?]feature=[^&]+)", "", url)
  return url.strip()


# --- UI設計 ---
st.title("🎵 YouTube BPM & Key Analyzer")
st.write(
    "YouTube動画のURLを入力するだけで、BPMとKey（調）を自動解析します（API中継モード）。"
)

url_input = st.text_input(
    "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
)

if st.button("解析開始", type="primary"):
  if not url_input:
    st.warning("YouTubeのURLを入力してください。")
  else:
    target_url = clean_youtube_url(url_input)
    with st.spinner("🔄 外部API中継経由で音声を取得して解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          # パブリックな音声抽出API（Cobalt等のオープンAPI互換エンドポイント）へリクエスト
          # クライアント側のIPではなくAPIサーバー側で処理されるためYouTubeのブロックを回避可能
          api_url = "https://co.wuk.sh/api/json"
          headers = {
              "Accept": "application/json",
              "Content-Type": "application/json",
          }
          payload = {
              "url": target_url,
              "downloadMode": "audio",
              "audioFormat": "mp3",
          }

          response = requests.post(
              api_url, json=payload, headers=headers, timeout=30
          )
          res_data = response.json()

          if res_data.get("status") in ["stream", "redirect", "picker"]:
            audio_download_url = res_data.get("url")
            title = res_data.get("filename", "Unknown Title")

            if not audio_download_url:
              # ピッカーなどの複数候補がある場合のフォールバック
              if (
                  "picker" in res_data
                  and len(res_data["picker"]) > 0
              ):
                audio_download_url = res_data["picker"][0].get("url")

            if not audio_download_url:
              raise Exception(
                  "音声のダウンロードリンクを取得できませんでした。"
              )

            # 実際の音声ファイルを一時ディレクトリにダウンロード
            audio_res = requests.get(
                audio_download_url, stream=True, timeout=60
            )
            audio_path = os.path.join(temp_dir, "downloaded_audio.mp3")
            with open(audio_path, "wb") as f:
              for chunk in audio_res.iter_content(chunk_size=8192):
                if chunk:
                  f.write(chunk)

            # librosaで読み込み（最初の90秒間を対象に高速解析）
            y, sr = librosa.load(audio_path, sr=22050, duration=90)

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

          else:
            err_msg = res_data.get(
                "text", "APIからのレスポンスが無効です。"
            )
            raise Exception(f"APIエラー: {err_msg}")

        except Exception as e:
          st.error(f"エラーが発生しました:\n`{e}`")
