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
    "AIが楽曲の構造変化（Aメロ・サビ等の境界）を自動検出し、区間ごとに高精度に解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
  st.audio(uploaded_file)

  if st.button("解析開始", type="primary"):
    with st.spinner("🤖 AIが楽曲の構造を解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          audio_path = os.path.join(temp_dir, uploaded_file.name)
          with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          # 音声全体の読み込み
          y, sr = librosa.load(audio_path, sr=22050)
          total_duration = librosa.get_duration(y=y, sr=sr)

          # 全体の代表値算出
          y_trimmed_full = y[: int(sr * min(total_duration, 60))]
          tempo_full, _ = librosa.beat.beat_track(y=y_trimmed_full, sr=sr)
          overall_bpm = (
              float(tempo_full[0])
              if isinstance(tempo_full, np.ndarray)
              else float(tempo_full)
          )
          overall_key = estimate_key(y, sr)

          # 結果表示（サマリー）
          st.success("解析完了！")
          st.subheader(f"ファイル名: {uploaded_file.name}")

          col1, col2 = st.columns(2)
          with col1:
            st.metric(label="全体平均 BPM", value=f"{overall_bpm:.1f}")
          with col2:
            st.metric(label="全体代表 Key", value=overall_key)

          st.divider()
          st.markdown("### 🤖 AI構造検出によるセクション別詳細")
          st.write(
              "音響的特徴の変化（盛り上がりやリズムの変化点）をAIが捉え、意味のある区間に分割して解析しています。"
          )

          # --- AIによる境界検出（構造解析） ---
          # MFCC（音色特徴量）を抽出して、曲の展開が変わるポイントを特定
          mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
          # テンポ・ビートに合わせた同期
          bound_frames = librosa.segment.agglomerative(mfcc, k=min(6, max(3, int(total_duration / 20))))
          bound_times = librosa.frames_to_time(bound_frames, sr=sr)

          # 境界の時間を綺麗にソートして、最初と最後（0秒と曲の長さ）を確実に入れる
          bound_times = np.unique(
              np.concatenate(([0.0], bound_times, [total_duration]))
          )
          bound_times.sort()

          # 各セクションごとの解析・表示
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

            # 短すぎる区間（2秒未満）はノイズとしてスキップ
            if (end_time - start_time) < 2.0:
              continue

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_y = y[start_sample:end_sample]

            # この区間のBPM
            chunk_tempo, _ = librosa.beat.beat_track(y=chunk_y, sr=sr)
            chunk_bpm = (
                float(chunk_tempo[0])
                if isinstance(chunk_tempo, np.ndarray)
                else float(chunk_tempo)
            )

            # この区間のKey
            chunk_key = estimate_key(chunk_y, sr)

            # セクション名の割り当て
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
                st.metric(
                    label="BPM",
                    value=f"{chunk_bpm:.1f}",
                    delta=f"{chunk_bpm - overall_bpm:.1f} (vs全体)"
                    if chunk_bpm != overall_bpm
                    else None,
                )
              with sc_col2:
                st.metric(label="Key", value=chunk_key)
              st.markdown("---")

          with st.expander("💡 AI構造解析についての解説"):
            st.caption(
                "音声のスペクトルや音色の変化（MFCC）を元に、機械学習の手法（凝集型クラスタリング）で"
                "「曲の雰囲気が変わる境界」を自動で割り出し、それぞれのブロックでBPMとKeyを計算しています。"
            )

        except Exception as e:
          st.error(f"解析エラーが発生しました:\n`{e}`")
