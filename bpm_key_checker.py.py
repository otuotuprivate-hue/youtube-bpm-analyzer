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
    "音楽ファイルをアップロードすると、セクション（展開）ごとにBPMとKeyを個別解析します。"
)

uploaded_file = st.file_uploader(
    "音声ファイルをアップロード", type=["mp3", "wav", "m4a", "flac", "ogg"]
)

if uploaded_file is not None:
  st.audio(uploaded_file)

  if st.button("解析開始", type="primary"):
    with st.spinner("🎼 楽曲のセクション（Aメロ・サビ等）を解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          audio_path = os.path.join(temp_dir, uploaded_file.name)
          with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          # 全体を読み込み
          y, sr = librosa.load(audio_path, sr=22050)
          total_duration = librosa.get_duration(y=y, sr=sr)

          # 全体の代表値（全体サマリー用）
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
          st.markdown("### 📊 セクション別（展開ごと）の詳細分析")
          st.write(
              "楽曲を一定のタイムライン（約15秒刻み）に分割し、それぞれの区間ごとのBPMとKeyの変化を追跡しています。"
          )

          # 15秒ごとのセクションに分割して個別に解析
          chunk_duration = 15.0  . # 1セクションあたりの秒数
          num_chunks = int(np.ceil(total_duration / chunk_duration))

          # セクション名ラベルの推測（大体の目安）
          section_labels = [
              "イントロ / Aメロ前半",
              "Aメロ後半 / Bメロ",
              "Bメロ / サビ移行",
              "サビ / メインパート",
              "間奏 / Cメロ",
              "落ちサビ / 大サビ",
              "アウトロ / 終盤",
          ]

          for i in range(min(num_chunks, 7)):  # 最大7セクションまで表示
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, total_duration)

            if start_time >= total_duration:
              break

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            chunk_y = y[start_sample:end_sample]

            if len(chunk_y) < sr * 2:  # 短すぎる断片はスキップ
              continue

            # このセクションのBPM
            chunk_tempo, _ = librosa.beat.beat_track(y=chunk_y, sr=sr)
            chunk_bpm = (
                float(chunk_tempo[0])
                if isinstance(chunk_tempo, np.ndarray)
                else float(chunk_tempo)
            )

            # このセクションのKey
            chunk_key = estimate_key(chunk_y, sr)

            # セクション名の決定
            label = (
                section_labels[i]
                if i < len(section_labels)
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

          with st.expander("💡 セクション解析についての解説"):
            st.caption(
                "楽曲内で転調（Keyの変更）やテンポの揺らぎ（ルバートや加速など）がある場合、"
                "このように細かく区切ることでどのパートで変化したかを視覚的に確認できます。"
            )

        except Exception as e:
          st.error(f"解析エラーが発生しました:\n`{e}`")
