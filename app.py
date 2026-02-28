from __future__ import annotations

import io
import threading

import pandas as pd
import streamlit as st

from batch_processor import BatchProcessor, TTSResult, TTSTask
from tts_client import TTSClient, TTSError
from utils import build_zip, parse_dataframe, parse_text_lines, safe_filename

st.set_page_config(page_title="字节 TTS 语音合成", page_icon="🎙️", layout="wide")

# ─────────────────────── 豆包 TTS 2.0 音色列表 ───────────────────────
VOICES_2 = {
    "通用场景": [
        ("Vivi 2.0", "zh_female_vv_uranus_bigtts"),
        ("小何 2.0", "zh_female_xiaohe_uranus_bigtts"),
        ("云舟 2.0", "zh_male_m191_uranus_bigtts"),
        ("小天 2.0", "zh_male_taocheng_uranus_bigtts"),
    ],
    "视频配音": [
        ("大壹", "zh_male_dayi_saturn_bigtts"),
        ("黑猫侦探社咪", "zh_female_mizai_saturn_bigtts"),
        ("鸡汤女", "zh_female_jitangnv_saturn_bigtts"),
        ("魅力女友", "zh_female_meilinvyou_saturn_bigtts"),
        ("流畅女声", "zh_female_santongyongns_saturn_bigtts"),
        ("儒雅逸辰", "zh_male_ruyayichen_saturn_bigtts"),
    ],
    "有声阅读": [
        ("儿童绘本", "zh_female_xueayi_saturn_bigtts"),
    ],
    "角色扮演": [
        ("可爱女生", "saturn_zh_female_keainvsheng_tob"),
        ("调皮公主", "saturn_zh_female_tiaopigongzhu_tob"),
        ("爽朗少年", "saturn_zh_male_shuanglangshaonian_tob"),
        ("天才同桌", "saturn_zh_male_tiancaitongzhuo_tob"),
        ("知性灿灿", "saturn_zh_female_cancan_tob"),
    ],
    "客服场景": [
        ("轻盈朵朵 2.0", "saturn_zh_female_qingyingduoduo_cs_tob"),
        ("温婉珊珊 2.0", "saturn_zh_female_wenwanshanshan_cs_tob"),
        ("热情艾娜 2.0", "saturn_zh_female_reqingaina_cs_tob"),
    ],
    "多语种": [
        ("Tim（美式英语）", "en_male_tim_uranus_bigtts"),
        ("Dacey（美式英语）", "en_female_dacey_uranus_bigtts"),
        ("Stokie（美式英语）", "en_female_stokie_uranus_bigtts"),
    ],
}

# 扁平列表：("场景 · 名称", voice_type)
VOICE_OPTIONS = [("自定义（手动输入）", "__custom__")]
for scene, items in VOICES_2.items():
    for name, vt in items:
        VOICE_OPTIONS.append((f"{scene} · {name}", vt))

VOICE_LABELS = [label for label, _ in VOICE_OPTIONS]
VOICE_MAP = {label: vt for label, vt in VOICE_OPTIONS}

app_id = "8834139548"
access_token = "Mh-NUv37j4Bh-BW53mQmq2DiglJ32BnY"
resource_id = "seed-tts-2.0"
max_workers = 10


def get_client() -> TTSClient | None:
    if not app_id or not access_token or not resource_id:
        st.warning("请先在侧边栏填写 App ID、Access Token 和 Resource ID")
        return None
    return TTSClient(app_id, access_token, resource_id)


def voice_selector(key_prefix: str) -> str:
    """用 pills 平铺音色选择，按场景分组。"""
    for scene, items in VOICES_2.items():
        names = [name for name, _ in items]
        vt_map = {name: vt for name, vt in items}
        sel_key = f"{key_prefix}_pills_{scene}"
        # 保持跨 scene 互斥：若其他 scene 刚被选中，清空本 scene
        chosen = st.pills(scene, names, key=sel_key)
        if chosen:
            # 清除其他 scene 的选中
            for other_scene in VOICES_2:
                if other_scene != scene:
                    other_key = f"{key_prefix}_pills_{other_scene}"
                    if st.session_state.get(other_key):
                        st.session_state[other_key] = None
            st.session_state[f"{key_prefix}_selected_vt"] = vt_map[chosen]

    # 自定义
    custom = st.text_input(
        "自定义音色 ID（填写后覆盖上方选择）",
        placeholder="例如：zh_female_shuangkuaisisi_moon_bigtts",
        key=f"{key_prefix}_custom_vt",
        label_visibility="collapsed",
    )
    if custom.strip():
        return custom.strip()
    vt = st.session_state.get(f"{key_prefix}_selected_vt", "zh_female_vv_uranus_bigtts")
    st.caption(f"当前音色：`{vt}`")
    return vt


# ─────────────────────── Tab 布局 ───────────────────────
tab1, tab2 = st.tabs(["🎵 单条合成", "📦 批量合成"])

# ══════════════════════ Tab 1: 单条合成 ══════════════════════
with tab1:
    st.header("单条语音合成")

    single_voice = voice_selector("single")

    text_input = st.text_area(
        "输入文本",
        height=150,
        placeholder="在此输入要合成的文本...",
        key="single_text",
    )

    st.markdown("**语音指令**（可选）")
    presets = {
        "😢 痛心": "你可以用特别特别痛心的语气说话吗?",
        "😄 欢乐": "嗯，你的语气再欢乐一点",
        "😤 骄傲": "你能用骄傲的语气来说话吗？",
        "🐢 说慢点": "你可以说慢一点吗？",
        "🔇 小声点": "你嗓门再小点。",
    }
    preset_cols = st.columns(len(presets))
    for col, (label, text) in zip(preset_cols, presets.items()):
        if col.button(label, key=f"preset_single_{label}"):
            st.session_state["context_texts_raw"] = text
    context_texts_raw = st.text_input(
        "语音指令",
        placeholder="可留空；例如：嗯，你的语气再欢乐一点",
        key="context_texts_raw",
        label_visibility="collapsed",
    )

    if st.button("🔊 开始合成", key="single_synthesize", type="primary"):
        if not text_input.strip():
            st.error("请输入文本")
        else:
            client = get_client()
            if client:
                context_texts = [context_texts_raw.strip()] if context_texts_raw.strip() else None

                with st.spinner("合成中，请稍候..."):
                    try:
                        audio_bytes = client.synthesize(
                            text=text_input.strip(),
                            voice_type=single_voice,
                            context_texts=context_texts,
                        )
                        st.session_state["single_result"] = {
                            "audio": audio_bytes,
                            "text": text_input.strip(),
                        }
                    except TTSError as e:
                        st.error(f"合成失败：{e}")
                        if e.status_code:
                            st.code(f"状态码: {e.status_code}\n{e.response_text or ''}")

    # 持久化显示结果（防止重刷丢失）
    if "single_result" in st.session_state:
        res = st.session_state["single_result"]
        st.success(f"合成成功！音频大小：{len(res['audio']) / 1024:.1f} KB")
        st.audio(res["audio"], format="audio/mp3")
        fname = safe_filename(res["text"], 0)
        st.download_button(
            label="⬇️ 下载 MP3",
            data=res["audio"],
            file_name=fname,
            mime="audio/mpeg",
            key="single_download",
        )

# ══════════════════════ Tab 2: 批量合成 ══════════════════════
with tab2:
    st.header("批量语音合成")

    batch_voice = voice_selector("batch")

    input_mode = st.radio(
        "输入方式",
        ["📄 Excel / CSV 上传", "📝 文本框多行输入"],
        horizontal=True,
        key="input_mode",
    )

    tasks: list[TTSTask] = []

    if input_mode == "📄 Excel / CSV 上传":
        st.caption("支持列名：`text/文本/内容` | `voice_type/音色`（可选，覆盖上方选择）| `context_texts/语音指令`（可选）")
        uploaded_file = st.file_uploader(
            "上传 Excel 或 CSV 文件",
            type=["xlsx", "xls", "csv"],
            key="uploaded_file",
        )
        if uploaded_file and batch_voice:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.dataframe(df.head(5), use_container_width=True)
                tasks = parse_dataframe(df, batch_voice)
                st.info(f"解析到 {len(tasks)} 条有效文本")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"文件解析失败：{e}")
    else:
        bulk_text = st.text_area(
            "每行一条文本（# 开头为注释，将跳过）",
            height=180,
            placeholder="第一条文本\n第二条文本\n# 这是注释行，会跳过\n第三条文本",
            key="bulk_text",
        )

        st.markdown("**语音指令**（可选，应用到所有条目）")
        batch_presets = {
            "😢 痛心": "你可以用特别特别痛心的语气说话吗?",
            "😄 欢乐": "嗯，你的语气再欢乐一点",
            "😤 骄傲": "你能用骄傲的语气来说话吗？",
            "🐢 说慢点": "你可以说慢一点吗？",
            "🔇 小声点": "你嗓门再小点。",
        }
        bp_cols = st.columns(len(batch_presets))
        for col, (label, text) in zip(bp_cols, batch_presets.items()):
            if col.button(label, key=f"preset_batch_{label}"):
                st.session_state["batch_context_texts"] = text
        batch_context_raw = st.text_input(
            "批量语音指令",
            placeholder="可留空；例如：嗯，你的语气再欢乐一点",
            key="batch_context_texts",
            label_visibility="collapsed",
        )

        if bulk_text.strip() and batch_voice:
            global_context = [batch_context_raw.strip()] if batch_context_raw.strip() else None
            tasks = parse_text_lines(bulk_text, batch_voice, global_context)
            st.info(f"解析到 {len(tasks)} 条有效文本" + (f"，语音指令：「{batch_context_raw.strip()}」" if global_context else ""))

    # 开始批量合成
    start_batch = st.button(
        "🚀 开始批量合成",
        disabled=len(tasks) == 0,
        type="primary",
        key="start_batch",
    )

    if start_batch:
        client = get_client()
        if client:
            st.session_state["batch_results"] = {}
            st.session_state["batch_tasks"] = tasks
            st.session_state["batch_running"] = True

    # ── 批量合成执行 & 展示 ──
    if st.session_state.get("batch_running"):
        tasks_to_run: list[TTSTask] = st.session_state["batch_tasks"]
        n = len(tasks_to_run)

        progress_bar = st.progress(0, text=f"0 / {n} 完成")
        result_slots = {i: st.empty() for i in range(n)}

        counter = [0]
        lock = threading.Lock()

        def on_result(result: TTSResult):
            with lock:
                st.session_state["batch_results"][result.index] = result
                counter[0] += 1
                progress = counter[0] / n
                progress_bar.progress(progress, text=f"{counter[0]} / {n} 完成")

            with result_slots[result.index].container():
                label = f"**#{result.index + 1}** {result.text[:50]}{'...' if len(result.text) > 50 else ''}"
                if result.success:
                    st.success(label)
                    st.audio(result.audio, format="audio/mp3")
                    st.download_button(
                        label="⬇️ 下载",
                        data=result.audio,
                        file_name=safe_filename(result.text, result.index),
                        mime="audio/mpeg",
                        key=f"dl_{result.index}",
                    )
                else:
                    st.error(f"{label}\n\n❌ 错误：{result.error}")

        processor = BatchProcessor(
            client=TTSClient(app_id, access_token, resource_id),
            max_workers=max_workers,
        )
        processor.process(tasks_to_run, on_result=on_result)

        st.session_state["batch_running"] = False
        progress_bar.progress(1.0, text=f"全部完成！{n} / {n}")

        # ZIP 下载
        results_list = [
            st.session_state["batch_results"].get(i)
            for i in range(n)
            if st.session_state["batch_results"].get(i)
        ]
        success_count = sum(1 for r in results_list if r and r.success)
        if success_count > 0:
            zip_buf = build_zip(results_list)
            st.download_button(
                label=f"📦 下载全部 ZIP（{success_count} 个文件）",
                data=zip_buf,
                file_name="tts_batch_output.zip",
                mime="application/zip",
                key="zip_download",
            )

    elif "batch_results" in st.session_state and st.session_state["batch_results"]:
        # 页面重刷后恢复显示
        results_map: dict[int, TTSResult] = st.session_state["batch_results"]
        tasks_list: list[TTSTask] = st.session_state.get("batch_tasks", [])
        n = len(tasks_list)

        st.success(f"上次批量合成结果（共 {n} 条）")
        for i in range(n):
            result = results_map.get(i)
            if result is None:
                continue
            label = f"**#{i + 1}** {result.text[:50]}{'...' if len(result.text) > 50 else ''}"
            if result.success:
                st.success(label)
                st.audio(result.audio, format="audio/mp3")
                st.download_button(
                    label="⬇️ 下载",
                    data=result.audio,
                    file_name=safe_filename(result.text, result.index),
                    mime="audio/mpeg",
                    key=f"dl_restored_{i}",
                )
            else:
                st.error(f"{label}\n\n❌ 错误：{result.error}")

        success_count = sum(1 for r in results_map.values() if r and r.success)
        if success_count > 0:
            zip_buf = build_zip(list(results_map.values()))
            st.download_button(
                label=f"📦 下载全部 ZIP（{success_count} 个文件）",
                data=zip_buf,
                file_name="tts_batch_output.zip",
                mime="application/zip",
                key="zip_download_restored",
            )
