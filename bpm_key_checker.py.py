import os
import tempfile
import librosa
import numpy as np
import streamlit as st

# --- ページ設定 ---
st.set_page_config(
    page_title="taetae-bpm-analyzer", page_icon="🎵", layout="centered"
)

# --- キー判定プロファイル ---
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


def time_to_seconds(time_str):
  """「1:30」や「01:30.5」などの文字列を秒数（float）に変換する"""
  time_str = time_str.strip()
  if not time_str:
    return 0.0

  try:
    return float(time_str)
  except ValueError:
    pass

  parts = time_str.split(":")
  if len(parts) == 2:
    try:
      return float(parts[0]) * 60 + float(parts[1])
    except ValueError as e:
      raise ValueError("時間形式が正しくありません（例: 1:30）") from e
  elif len(parts) == 3:
    try:
      return (
          float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
      )
    except ValueError as e:
      raise ValueError("時間形式が正しくありません（例: 1:02:30）") from e
  else:
    raise ValueError("時間形式が正しくありません（例: 1:30）")


def analyze_all_keys(y, sr):
  chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
  chroma_sum = np.sum(chroma, axis=1)
  if np.sum(chroma_sum) > 0:
    chroma_sum = chroma_sum / np.sum(chroma_sum)

  key_results = []
  for i in range(12):
    maj_prof = np.roll(MAJOR_PROFILE, i)
    min_prof = np.roll(MINOR_PROFILE, i)

    corr_maj = np.corrcoef(chroma_sum, maj_prof)[0, 1]
    corr_min = np.corrcoef(chroma_sum, min_prof)[0, 1]

    key_results.append({
        "key": f"{NOTE_NAMES[i]} Major ({NOTE_NAMES[i]})",
        "score": float(corr_maj) if not np.isnan(corr_maj) else 0.0,
    })
    key_results.append({
        "key": f"{NOTE_NAMES[i]} Minor ({NOTE_NAMES[i]}m)",
        "score": float(corr_min) if not np.isnan(corr_min) else 0.0,
    })

  key_results = sorted(key_results, key=lambda x: x["score"], reverse=True)
  raw_scores = [max(0.0, item["score"]) for item in key_results]
  total_score = sum(raw_scores)

  if total_score > 0:
    for item, raw in zip(key_results, raw_scores):
      item["proportion"] = (raw / total_score) * 100
  else:
    for item in key_results:
      item["proportion"] = 0.0

  return key_results


def estimate_overall_bpm(y, sr):
  _, y_percussive = librosa.effects.hpss(y, margin=3.0)
  onset_env = librosa.onset.onset_strength(
      y=y_percussive, sr=sr, hop_length=512, aggregate=np.median
  )
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

  if 112 <= bpm <= 122:
    corrected = bpm * 1.5
    if 165 <= corrected <= 185:
      bpm = corrected

  while bpm < 75:
    bpm *= 2
  while bpm > 185:
    bpm /= 2

  return round(bpm, 1)


# --- UI設計 ---
st.title("🎵 taetae-bpm-analyzer")
st.write(
    "音楽ファイルをアップロードすると、BPMと全キーの適合割合を解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
  st.audio(uploaded_file)

  st.markdown("### ⚙️ 解析範囲の設定")
  full_range = st.checkbox(
      "曲の最後まで（フルで）解析する",
      value=True,
      help="チェックを外すと、指定した区間のみを解析します。",
  )

  col1_ui, col2_ui = st.columns(2)
  with col1_ui:
    offset_str = st.text_input("開始位置 (例: 0:00 または 30)", value="0:00")

  end_str = None
  if not full_range:
    with col2_ui:
      end_str = st.text_input("終了位置 (例: 1:30 または 90)", value="1:30")

  if st.button("解析開始", type="primary"):
    with st.spinner("🎧 解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          try:
            offset_sec = time_to_seconds(offset_str)
          except ValueError as ve:
            st.error(f"開始位置エラー: {ve}")
            st.stop()

          audio_path = os.path.join(temp_dir, uploaded_file.name)
          with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          if full_range or not end_str:
            y, sr = librosa.load(
                audio_path, sr=22050, offset=offset_sec, duration=None
            )
            range_desc = f"（解析範囲: {offset_str} 〜 最後）"
          else:
            try:
              end_sec = time_to_seconds(end_str)
            except ValueError as ve:
              st.error(f"終了位置エラー: {ve}")
              st.stop()

            if end_sec <= offset_sec:
              st.error("エラー: 終了位置は開始位置より後にしてください。")
              st.stop()

            y, sr = librosa.load(
                audio_path,
                sr=22050,
                offset=offset_sec,
                duration=(end_sec - offset_sec),
            )
            range_desc = f"（解析範囲: {offset_str} 〜 {end_str}）"

          bpm = estimate_overall_bpm(y, sr)
          key_rankings = analyze_all_keys(y, sr)
          best_key = key_rankings[0]["key"]

          st.success("解析完了！")
          st.subheader(f"ファイル名: {uploaded_file.name}")
          st.caption(range_desc)

          c1, c2 = st.columns(2)
          with c1:
            st.metric(label="検出 BPM", value=f"{bpm:.1f}")
          with c2:
            st.metric(label="検出 Key（最有力）", value=best_key)

          st.divider()
          st.markdown("### 🎹 全24キーの適合割合")
          for rank, item in enumerate(key_rankings, 1):
            if rank == 1:
              st.markdown(
                  f"**🥇 1位: {item['key']}** — 割合: **{item['proportion']:.1f}%**"
                  f" (スコア: {item['score']:.3f})"
              )
            elif rank <= 5:
              st.markdown(
                  f"🥈 {rank}位: {item['key']} — 割合:"
                  f" {item['proportion']:.1f}% (スコア: {item['score']:.3f})"
              )

          with st.expander("📋 全24キーのスコア詳細をすべて見る"):
            for rank, item in enumerate(key_rankings, 1):
              st.text(
                  f"{rank:2d}位: {item['key']} | 割合: {item['proportion']:5.1f}%"
                  f" | スコア: {item['score']:.3f}"
              )

          with st.expander("💡 テンポ（BPM）の微調整用"):
            st.write(
                f"- **通常**: `{bpm:.1f}`\n- **1.5倍**: `{bpm * 1.5:.1f}`\n-"
                f" **倍テンポ**: `{bpm * 2:.1f}`\n- **半テンポ**: `{bpm / 2:.1f}`"
            )

        except Exception as e:
          st.error(f"解析エラーが発生しました:\n`{e}`")
