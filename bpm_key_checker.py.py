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


def analyze_all_keys(y, sr):
    """全24キーの相関を計算し、確率（割合）を算出してランキング形式で返す"""
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

    # スコア（相関係数）を降順ソート
    key_results = sorted(key_results, key=lambda x: x["score"], reverse=True)

    # 正の相関値を抽出して割合（％）に正規化
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

    # 3連符や裏拍の誤認（約1.5倍ズレ）の自動補正
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

  if st.button("解析開始", type="primary"):
    with st.spinner("🎧 BPMと全キーの適合度を解析中..."):
      with tempfile.TemporaryDirectory() as temp_dir:
        try:
          audio_path = os.path.join(temp_dir, uploaded_file.name)
          with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

          y, sr = librosa.load(audio_path, sr=22050)

          bpm = estimate_overall_bpm(y, sr)
          key_rankings = analyze_all_keys(y, sr)
          best_key = key_rankings[0]["key"]

          double_bpm = round(bpm * 2, 1)
          half_bpm = round(bpm / 2, 1)

          st.success("解析完了！")
          st.subheader(f"ファイル名: {uploaded_file.name}")

          col1, col2 = st.columns(2)
          with col1:
            st.metric(label="検出 BPM", value=f"{bpm:.1f}")
          with col2:
            st.metric(label="検出 Key（最有力）", value=best_key)

          # 全キーの割合表示セクション
          st.divider()
          st.markdown("### 🎹 全24キーの適合割合（スコアランキング）")
          st.write(
              "楽曲のコード成分が各調（メジャー/マイナー）の理論プロファイルとどのくらい一致しているかの割合です。"
          )

          # 上位5位までをピックアップしてプログレスバー等で視覚的に表示
          for rank, item in enumerate(key_rankings, 1):
            k_name = item["key"]
            prop = item["proportion"]
            score = item["score"]

            # 上位のキーはハイライト表示
            if rank == 1:
              st.markdown(
                  f"**🥇 1位: {k_name}** — 割合: **{prop:.1f}%** (相関スコア:"
                  f" {score:.3f})"
              )
            elif rank <= 5:
              st.markdown(
                  f"🥈 {rank}位: {k_name} — 割合: {prop:.1f}% (スコア:"
                  f" {score:.3f})"
              )
            else:
              with st.expander("📉 6位以下の詳細スコア一覧"):
                # 残りをまとめてテーブル/リスト表示
                pass

          # 6位以下をまとめて見られるセクション
          with st.expander("📋 全24キーのスコア詳細をすべて見る"):
            for rank, item in enumerate(key_rankings, 1):
              st.text(
                  f"{rank:2d}位: {item['key']} | 割合: {item['proportion']:5.1f}%"
                  f" | スコア: {item['score']:.3f}"
              )

          with st.expander("💡 テンポ（BPM）の微調整用"):
            st.write(
                f"- **通常候補**: `{bpm:.1f}`\n"
                f"- **1.5倍補正候補**: `{bpm * 1.5:.1f}`\n"
                f"- **倍テンポ候補**: `{double_bpm:.1f}`\n"
                f"- **半テンポ候補**: `{half_bpm:.1f}`"
            )

        except Exception as e:
          st.error(f"解析エラーが発生しました:\n`{e}`")
