# plagiarism_detector.py
import os, time, logging, uuid, requests
from werkzeug.utils import secure_filename
from pefile import PE
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
from binary_analyzer import BinaryComparator, AIModelAnalyzer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class EnhancedAIModelAnalyzer(AIModelAnalyzer):
    def __init__(self, model_type, api_key):
        super().__init__(model_type, api_key)
        self.session = self._create_http_session()
        self.timeout = (600, 600)

    def _create_http_session(self):
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def generate_analysis(self, code1, code2, score):
        prompt = f"""你是一名专业的二进制代码安全分析师，下面为你提供两份二进制代码的反汇编片段以及它们的相似度得分，请依据这些信息从功能相似性、安全风险、混淆迹象、架构差异四个维度进行详细分析。
相似度得分：{score:.2f}
代码片段1特征：{code1}
代码片段2特征：{code2}
请按照以下格式输出分析结果：
功能相似性：[详细分析]
安全风险：[详细分析]
混淆迹象：[详细分析]
架构差异：[详细分析]"""
        for attempt in range(3):
            try:
                response = self.session.post(
                    "https://api.dashscope.com/v1/generation/text",
                    json={
                        "model": self.model_type,
                        "prompt": prompt,
                        "max_length": 2000,
                        "temperature": 0.7
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["output"]["text"]
            except Exception as e:
                logging.warning(f"模型请求失败：{e}")
        return None

def detect_architecture(file_path):
    try:
        pe = PE(file_path)
        return CS_MODE_32 if pe.FILE_HEADER.Machine == 0x014c else CS_MODE_64
    except Exception:
        return CS_MODE_32

def disassemble_binary(file_path):
    mode = detect_architecture(file_path)
    md = Cs(CS_ARCH_X86, mode)
    md.detail = True
    with open(file_path, 'rb') as f:
        code = f.read()
    return "\n".join(f"{i.address:08x}: {i.mnemonic}\t{i.op_str}" for i in md.disasm(code[:1024], 0x1000)) or None

def save_uploaded_files(files, upload_dir):
    if len(files) < 2:
        return None, "请至少上传两个文件"
    file_paths = []
    for file in files:
        if not file.filename:
            return None, "文件名不能为空"
        filename = secure_filename(file.filename)
        path = os.path.join(upload_dir, filename)
        file.save(path)
        file_paths.append(path)
    return file_paths, None

def delete_uploaded_files(file_paths):
    for path in file_paths:
        for _ in range(3):
            try:
                os.remove(path)
                break
            except PermissionError:
                time.sleep(1)

def run_plagiarism_check(comparator, files, api_key, model_type, upload_dir):
    file_paths, err = save_uploaded_files(files, upload_dir)
    if err:
        return None, err

    try:
        sig1 = comparator._get_code_signature(file_paths[0])
        sig2 = comparator._get_code_signature(file_paths[1])
        asm1 = disassemble_binary(file_paths[0])
        asm2 = disassemble_binary(file_paths[1])
        if not sig1 or not sig2 or not asm1 or not asm2:
            raise ValueError("签名或反汇编失败")

        lexical_sim = comparator._lexical_similarity(sig1['code'], sig2['code'])
        semantic_sim = comparator._calculate_similarity(sig1, sig2)
        syntactic_sim = comparator._edit_similarity(sig1['code'], sig2['code'])
        combined = (lexical_sim + semantic_sim + syntactic_sim) / 3

        analyzer = EnhancedAIModelAnalyzer(model_type, api_key)
        report = analyzer.generate_analysis(asm1, asm2, combined)
        return {
            "lexical_similarity": lexical_sim,
            "semantic_similarity": semantic_sim,
            "syntactic_similarity": syntactic_sim,
            "combined_similarity": combined,
            "analysis_result": report,
            "disassembly_sample1": asm1[:200] + "...",
            "disassembly_sample2": asm2[:200] + "..."
        }, None
    except Exception as e:
        return None, f"检测失败: {str(e)}"
    finally:
        delete_uploaded_files(file_paths)
