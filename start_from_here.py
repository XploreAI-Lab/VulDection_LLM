from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import logging
from binary_analyzer import BinaryComparator, AIModelAnalyzer
from flask_cors import CORS
import time
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
import uuid
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your-secret-key-here'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 配置日志
logging.basicConfig(level=logging.DEBUG)

comparator = BinaryComparator("D:\\Program\\CodeT5\\codet5")

# 全局变量用于保存API密钥和模型类型
global_api_key = None
global_model_type = None


def allowed_file(filename):
    """验证文件类型"""
    ALLOWED_EXTENSIONS = {'exe', 'dll', 'elf', 'bin'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_api_key_and_model_type():
    """验证API密钥和模型类型"""
    global global_api_key, global_model_type
    api_key = global_api_key
    model_type = global_model_type

    if not api_key or not model_type:
        logging.error("缺少 API 密钥或模型类型")
        return None, None, {"error": "缺少 API 密钥或模型类型"}, 400

    return api_key, model_type, None, None


def save_uploaded_files(request):
    """保存上传的文件"""
    files = request.files.getlist('files')
    logging.debug(f"接收到的文件数量: {len(files)}")
    logging.debug(f"接收到的文件名称: {[file.filename for file in files]}")

    if len(files) < 2:
        logging.error("未找到足够的文件，请至少上传两个文件")
        return None, {"error": "未找到足够的文件，请至少上传两个文件"}, 400

    file_paths = []
    for file in files:
        if file.filename == '':
            logging.error("文件名不能为空")
            return None, {"error": "文件名不能为空"}, 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"保存文件: {file_path}")
        file.save(file_path)
        file_paths.append(file_path)

    return file_paths, None, None


def delete_uploaded_files(file_paths):
    """删除上传的文件"""
    for file_path in file_paths:
        max_retries = 3
        retries = 0
        time.sleep(2)
        while retries < max_retries:
            try:
                logging.info(f"删除文件: {file_path}")
                os.remove(file_path)
                break
            except PermissionError as e:
                retries += 1
                if retries < max_retries:
                    logging.warning(f"文件 {file_path} 被占用，等待 1 秒后重试（第 {retries} 次）")
                    time.sleep(1)
                else:
                    logging.error(f"无法删除文件 {file_path}: {str(e)}")


def detect_architecture(file_path):
    """检测二进制文件的架构（32位/64位）"""
    try:
        pe = pefile.PE(file_path)
        machine_type = pe.FILE_HEADER.Machine
        if machine_type == 0x014c:
            return CS_MODE_32
        elif machine_type == 0x8664:
            return CS_MODE_64
        else:
            logging.warning(f"未知PE架构: {hex(machine_type)}")
            return CS_MODE_32
    except Exception as e:
        logging.warning(f"无法检测架构，默认使用32位: {e}")
        return CS_MODE_32


def disassemble_binary(file_path):
    """使用Capstone反汇编二进制文件"""
    try:
        mode = detect_architecture(file_path)
        md = Cs(CS_ARCH_X86, mode)
        md.detail = True

        with open(file_path, 'rb') as f:
            code = f.read()

        disassembly = []
        count = 0
        for i in md.disasm(code[:1024], 0x1000):
            disassembly.append(f"{i.address:08x}: {i.mnemonic}\t{i.op_str}")
            count += 1
            if count >= 100:
                break

        if not disassembly:
            logging.warning("未能反汇编任何指令")
            return None

        return "\n".join(disassembly)
    except Exception as e:
        logging.error(f"Capstone反汇编失败: {str(e)}")
        return None


class EnhancedAIModelAnalyzer(AIModelAnalyzer):
    """增强型大模型分析器，支持超时重试和请求优化"""

    def __init__(self, model_type, api_key):
        super().__init__(model_type, api_key)
        self.session = self._create_http_session()
        self.timeout = (600, 600)

    def _create_http_session(self):
        """创建带重试机制的HTTP会话"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def generate_analysis(self, code1, code2, score):
        """优化大模型请求，限制输入长度并实现重试"""
        code1 = code1[:4000] if code1 else ""
        code2 = code2[:4000] if code2 else ""

        prompt = f"""你是一名专业的二进制代码安全分析师，下面为你提供两份二进制代码的反汇编片段以及它们的相似度得分，请依据这些信息从功能相似性、安全风险、混淆迹象、架构差异四个维度进行详细分析。
相似度得分：{score:.2f}
代码片段1特征：{code1}
代码片段2特征：{code2}
请按照以下格式输出分析结果：
功能相似性：[详细分析]
安全风险：[详细分析]
混淆迹象：[详细分析]
架构差异：[详细分析]
"""
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
            except requests.Timeout:
                if attempt < 2:
                    logging.warning(f"大模型请求超时，第{attempt + 1}次重试...")
                    time.sleep(30)
                else:
                    raise
            except Exception as e:
                logging.error(f"大模型API错误: {str(e)}")
                return None


@app.route('/')
def index():
    return render_template('index.html')

# 修复：添加所有页面的路由
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/analysis-results.html')
def analysis_results():
    return render_template('analysis-results.html')

@app.route('/vulnerable-detection.html')
def vulnerable_detection():
    return render_template('vulnerable-detection.html')

@app.route('/knowledge.html')
def knowledge():
    return render_template('knowledge.html')

@app.route('/settings.html')
def settings():
    return render_template('settings.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        api_key, model_type, error_response, error_status = validate_api_key_and_model_type()
        if error_response:
            return jsonify(error_response), error_status

        file_paths, error_response, error_status = save_uploaded_files(request)
        if error_response:
            return jsonify(error_response), error_status

        sig1 = comparator._get_code_signature(file_paths[0])
        sig2 = comparator._get_code_signature(file_paths[1])

        if sig1 is None or sig2 is None:
            logging.error("无法获取文件签名")
            delete_uploaded_files(file_paths)
            return jsonify({"error": "无法获取文件签名"}), 400

        asm_code1 = disassemble_binary(file_paths[0])
        asm_code2 = disassemble_binary(file_paths[1])

        if asm_code1 is None or asm_code2 is None:
            logging.error("无法反汇编二进制文件")
            delete_uploaded_files(file_paths)
            return jsonify({"error": "无法反汇编二进制文件，请检查文件格式"}), 400

        lexical_sim = comparator._lexical_similarity(sig1['code'], sig2['code'])
        semantic_sim = comparator._calculate_similarity(sig1, sig2)
        syntactic_sim = comparator._edit_similarity(sig1['code'], sig2['code'])

        combined_similarity = (lexical_sim + semantic_sim + syntactic_sim) / 3

        analyzer = EnhancedAIModelAnalyzer(model_type, api_key)
        try:
            start_time = time.time()
            analysis_result = analyzer.generate_analysis(asm_code1, asm_code2, combined_similarity)
            end_time = time.time()
            logging.info(f"大模型分析耗时: {end_time - start_time} 秒")

            if analysis_result is None:
                analysis_result = "大模型响应为空，请检查API密钥或网络连接"
        except Exception as e:
            logging.error(f"大模型分析失败: {str(e)}")
            analysis_result = f"大模型分析失败: {str(e)}，请检查网络连接或稍后重试"

        delete_uploaded_files(file_paths)

        return jsonify({
            "lexical_similarity": lexical_sim,
            "semantic_similarity": semantic_sim,
            "syntactic_similarity": syntactic_sim,
            "combined_similarity": combined_similarity,
            "analysis_result": analysis_result,
            "disassembly_sample1": asm_code1[:200] + "...",
            "disassembly_sample2": asm_code2[:200] + "..."
        })
    except Exception as e:
        logging.error(f"处理请求时出错: {str(e)}")
        return jsonify({"error": f"处理请求时出错: {str(e)}，请稍后重试"}), 500


@app.route('/save_settings', methods=['POST'])
def save_settings():
    global global_api_key, global_model_type
    data = request.get_json()

    if not data or 'apiKey' not in data or 'modelType' not in data:
        return jsonify({"success": False, "error": "缺少必要参数"}), 400

    api_key = data['apiKey'].strip()
    model_type = data['modelType'].strip()

    if not api_key.startswith('sk-') or len(api_key) < 20:
        return jsonify({"success": False, "error": "无效的API密钥格式"}), 400

    global_api_key = api_key
    global_model_type = model_type

    return jsonify({"success": True, "message": "配置保存成功"}), 200


@app.route('/api/get-ai-report', methods=['POST'])
def get_ai_report():
    if 'files' not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({"error": "至少需要两个文件进行AI分析"}), 400

    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"error": f"文件 {file.filename} 类型不允许"}), 400

    api_key, model_type, error_response, error_status = validate_api_key_and_model_type()
    if error_response:
        return error_response, error_status

    ai_analyzer = EnhancedAIModelAnalyzer(model_type=model_type, api_key=api_key)

    session_id = str(uuid.uuid4())

    work_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        file_paths = []
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(work_dir, filename)
            file.save(file_path)
            file_paths.append(file_path)

        disassembled_codes = []
        for path in file_paths:
            code = comparator.get_disassembled_code(path)
            if code is None:
                code = disassemble_binary(path)
            disassembled_codes.append(code)

        if any(code is None for code in disassembled_codes):
            raise ValueError("无法获取有效的反汇编代码")

        similarity_results = comparator.compare_files(file_paths, top_k=10)
        if not similarity_results:
            raise ValueError("无法计算文件相似度")

        top_similarity = similarity_results[0][2]

        report = ai_analyzer.generate_analysis(
            code1=disassembled_codes[0],
            code2=disassembled_codes[1],
            score=top_similarity
        )

        return jsonify({
            "status": "success",
            "report": report or "大模型分析超时，请重试",
            "similarity": round(top_similarity, 4),
            "session_id": session_id
        })

    except ValueError as ve:
        logging.error(f"AI报告生成失败: {str(ve)}")
        return jsonify({"error": f"分析失败: {str(ve)}"}), 400
    except Exception as e:
        logging.error(f"生成AI报告失败: {str(e)}")
        return jsonify({"error": "生成AI报告失败，请检查文件格式或联系管理员"}), 500
    finally:
        try:
            delete_uploaded_files(file_paths)
        except Exception as cleanup_e:
            logging.warning(f"文件清理失败: {str(cleanup_e)}")


if __name__ == '__main__':
    app.run(debug=True)