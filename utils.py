from __future__ import annotations

import io
import re
import zipfile

import pandas as pd

from batch_processor import TTSResult, TTSTask


# 列名别名映射
_TEXT_COLS = ["#文本", "text", "文本", "内容", "文字", "句子", "query", "问题"]
_VOICE_COLS = ["voice_type", "voice", "音色", "音色类型", "情感/状态类型"]
_CONTEXT_COLS = ["#语音指令", "context_texts", "context", "语音指令", "指令"]
_SECTION_COLS = ["section_id", "section", "会话ID", "分段ID"]


def strip_action_text(text: str) -> str:
    """去掉【】内的动作描述，只保留对白文本。"""
    cleaned = re.sub(r'【[^】]*】', '', text)
    return cleaned.strip()


def safe_filename(text: str, index: int) -> str:
    """生成安全的文件名：001_文本前20字.mp3"""
    clean = re.sub(r'[\\/:*?"<>|\n\r\t]', "", text)
    clean = clean.strip()[:20]
    return f"{index + 1:03d}_{clean}.mp3"


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
        # 大小写不敏感
        for col in df.columns:
            if col.strip().lower() == c.lower():
                return col
    return None


def _parse_context(val) -> list[str] | None:
    if pd.isna(val) or str(val).strip() == "":
        return None
    raw = str(val)
    # 支持逗号或换行分隔
    parts = re.split(r"[,，\n]+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else None


def parse_dataframe(df: pd.DataFrame, default_voice_type: str) -> list[TTSTask]:
    """解析 DataFrame，兼容多种列名，返回 TTSTask 列表。"""
    text_col = _find_col(df, _TEXT_COLS)
    if text_col is None:
        raise ValueError(
            f"找不到文本列，请确保 Excel/CSV 包含以下列名之一：{_TEXT_COLS}"
        )

    voice_col = _find_col(df, _VOICE_COLS)
    context_col = _find_col(df, _CONTEXT_COLS)
    section_col = _find_col(df, _SECTION_COLS)

    tasks = []
    for i, row in df.iterrows():
        text = strip_action_text(str(row[text_col]).strip())
        if not text or text.lower() == "nan":
            continue

        voice = default_voice_type
        if voice_col and not pd.isna(row.get(voice_col, None)):
            v = str(row[voice_col]).strip()
            if v:
                voice = v

        context_texts = None
        if context_col:
            context_texts = _parse_context(row.get(context_col))

        section_id = None
        if section_col and not pd.isna(row.get(section_col, None)):
            s = str(row[section_col]).strip()
            if s and s.lower() != "nan":
                section_id = s

        tasks.append(
            TTSTask(
                index=len(tasks),
                text=text,
                voice_type=voice,
                context_texts=context_texts,
                section_id=section_id,
            )
        )

    return tasks


def parse_text_lines(
    text: str,
    voice_type: str,
    context_texts: list[str] | None = None,
    section_id: str | None = None,
) -> list[TTSTask]:
    """解析多行文本，# 开头为注释行，跳过空行。context_texts/section_id 应用到所有条目。"""
    tasks = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(
            TTSTask(
                index=len(tasks),
                text=line,
                voice_type=voice_type,
                context_texts=context_texts,
                section_id=section_id,
            )
        )
    return tasks


def build_zip(results: list[TTSResult]) -> io.BytesIO:
    """将成功的合成结果打包为内存 ZIP，不落盘。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            if result.success:
                filename = safe_filename(result.text, result.index)
                zf.writestr(filename, result.audio)
    buf.seek(0)
    return buf
