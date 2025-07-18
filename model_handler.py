import requests
import json
import logging
import time
from typing import Optional


class ModelHandler:
    """统一大模型调用接口（严格遵循DeepSeek官方规范）"""

    def __init__(self, api_key: str, default_model: str = "deepseek"):
        self.api_key = api_key.strip()  # 增加密钥清理
        self.default_model = default_model
        self.base_url = {
            "deepseek": "https://api.deepseek.com/v1/chat/completions",
            "qwen": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        }
        self.timeout = 60
        self.max_retries = 2
        self._api_call_count = 0

    def generate_analysis(self, prompt: str, model_type: Optional[str] = None) -> str:
        """生成分析报告（增强错误处理）"""
        model = model_type or self.default_model
        try:
            if model == "deepseek":
                return self._call_deepseek_v2(prompt)  # 使用新版调用方法
            elif model == "qwen":
                return self._call_qwen(prompt)
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            logging.error(f"模型调用失败: {str(e)}", exc_info=True)
            return "分析服务暂不可用"

    # 新增深度求索官方规范调用方法
    def _call_deepseek_v2(self, prompt: str) -> str:
        """严格遵循DeepSeek最新接口规范"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Accept": "application/json"  # 明确声明接受的响应类型
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.3,
            "top_p": 0.95,  # 新增官方推荐参数
            "max_tokens": 2000,
            "stream": False  # 明确关闭流式传输
        }

        for attempt in range(self.max_retries + 1):  # 包含初始尝试
            try:
                response = requests.post(
                    self.base_url["deepseek"],
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                # 优先解析响应内容
                response_data = response.json()
                logging.debug(f"完整响应: {json.dumps(response_data, ensure_ascii=False)}")

                # 处理业务逻辑错误
                if "error" in response_data:
                    err_msg = response_data["error"].get("message", "未知错误")
                    err_code = response_data["error"].get("code", "unknown")
                    raise Exception(f"[{err_code}] {err_msg}")

                # 提取有效内容
                if response_data.get("choices"):
                    return response_data["choices"][0]["message"]["content"]

                # 若未触发上述条件则检查HTTP状态码
                response.raise_for_status()

            except requests.exceptions.HTTPError as e:
                # 特殊处理401错误
                if e.response.status_code == 401:
                    error_info = e.response.json().get("error", {})
                    raise Exception(f"认证失败: {error_info.get('message', '请检查API密钥和权限')}")

                # 其他HTTP错误
                if attempt == self.max_retries:
                    raise Exception(f"HTTP错误: {e.response.status_code} {e.response.reason}")

            except json.JSONDecodeError:
                logging.error(f"响应解析失败 | 原始内容: {response.text[:200]}")
                raise Exception("服务返回了无效的响应格式")

            except Exception as e:
                if attempt == self.max_retries:
                    raise e

            time.sleep(1.5 ** attempt)  # 指数退避策略

        return "请求失败，请稍后重试"

    def _call_qwen(self, prompt: str) -> str:
        """保持原有Qwen调用逻辑"""
        from dashscope import Generation
        Generation.api_key = self.api_key

        try:
            response = Generation.call(
                model='qwen-max',
                prompt=prompt,
                max_length=2000,
                temperature=0.3
            )
            if response.status_code != 200:
                raise ConnectionError(f"API请求失败: {response.status_code}")
            return response.output.text
        except Exception as e:
            raise RuntimeError(f"Qwen调用失败: {str(e)}")