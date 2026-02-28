from __future__ import annotations

import base64
import json
import requests


class TTSError(Exception):
    def __init__(self, message, status_code=None, response_text=None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class TTSClient:
    def __init__(self, app_id: str, access_token: str, resource_id: str):
        self.app_id = app_id
        self.access_token = access_token
        self.resource_id = resource_id
        self.base_url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"

    def synthesize(
        self,
        text: str,
        voice_type: str,
        context_texts: list[str] | None = None,
        section_id: str | None = None,
    ) -> bytes:
        """合成语音，返回 MP3 字节数据。失败时抛出 TTSError。"""
        headers = {
            "X-Api-App-Id": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "Content-Type": "application/json",
        }

        payload = {
            "namespace": "BidirectionalTTS",
            "req_params": {
                "text": text,
                "speaker": voice_type,
                "audio_params": {
                    "format": "mp3",
                    "sample_rate": 24000,
                },
            },
        }

        additions = {}
        if context_texts:
            additions["context_texts"] = context_texts
        if section_id:
            additions["section_id"] = section_id
        if additions:
            payload["req_params"]["additions"] = json.dumps(additions)

        try:
            session = requests.Session()
            response = session.post(
                self.base_url, headers=headers, json=payload, stream=True, timeout=60
            )
        except requests.RequestException as e:
            raise TTSError(f"网络请求失败: {e}") from e

        if response.status_code != 200:
            raise TTSError(
                f"API 返回错误状态码: {response.status_code}",
                status_code=response.status_code,
                response_text=response.text[:500],
            )

        audio_data = b""
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # 跳过心跳行或非 JSON 行
                    continue
                # 音频在 data 字段（v3 API）
                chunk = data.get("data")
                if chunk:
                    audio_data += base64.b64decode(chunk)
                # 检查 API 级别错误（非 0 且非结束码 20000000）
                code = data.get("code", 0)
                if code not in (0, 20000000) and code:
                    raise TTSError(
                        f"API 错误: code={code}, message={data.get('message', '')}",
                        status_code=code,
                    )
        except TTSError:
            raise
        except Exception as e:
            raise TTSError(f"解析响应时出错: {e}") from e

        if not audio_data:
            raise TTSError("API 返回了空的音频数据")

        return audio_data
