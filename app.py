# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid
import logging
from threading import Lock
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import atexit
import re

from binary_analyzer import BinaryComparator
from plagiarism_detector import run_plagiarism_check
from vulnerability_detector import VulnerabilityAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置HF镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
comparator = BinaryComparator("D:\\program\\CodeT5\\codet5")

# 全局变量：可由前端设置
global_api_key = None
global_model_type = None

# 知识库相关全局变量
embed_model = None
embedding_dim = 384  # MiniLM 默认维度
kb_index = None
kb_mapping = {}
knowledge_base_path = ""
rag_service_active = False
knowledge_lock = Lock()

def cleanup_resources():
    global kb_index, kb_mapping
    if kb_index:
        kb_index.reset()
        kb_index = None
        kb_mapping = {}
    logger.info("资源清理完成")

atexit.register(cleanup_resources)

def initialize_default_knowledge():
    global kb_index, kb_mapping
    # 扩展默认知识库内容
    default_kb_texts = [
        "缓冲区溢出是一种常见的内存操作漏洞，攻击者通过覆盖超出分配空间的数据来劫持程序控制流。",
        "格式化字符串漏洞允许攻击者读取或写入任意内存位置，可能导致敏感信息泄露或代码执行。",
        "整数溢出发生在算术运算结果超出变量类型表示范围时，可能导致缓冲区溢出或其他意外行为。",
        "堆溢出漏洞利用堆内存分配机制进行攻击，通常比栈溢出更复杂但危害性更大。",
        "释放后使用(UAF)漏洞发生在程序继续使用已释放的内存时，是现代浏览器和操作系统中常见的高危漏洞。",
        "竞争条件漏洞发生在多个线程或进程同时访问共享资源时，可能导致数据损坏或权限提升。",
        "符号执行是一种程序分析技术，通过探索程序的所有可能执行路径来发现潜在漏洞。",
        "模糊测试(Fuzzing)是一种自动化漏洞挖掘技术，通过向程序输入大量随机或半随机数据来触发异常。",
        "ROP(Return-Oriented Programming)是一种高级利用技术，通过组合现有代码片段(gadgets)来绕过DEP保护。",
        "ASLR(地址空间布局随机化)是现代操作系统采用的安全机制，通过随机化内存地址增加漏洞利用难度。",
        "Canary是一种栈保护机制，在函数返回地址前放置特殊值，用于检测栈溢出攻击。",
        "控制流完整性(CFI)是一种安全机制，确保程序执行流程不会偏离预期路径。",
        "逆向工程中常用工具包括IDA Pro、Ghidra、Radare2和Binary Ninja，各有其优势和使用场景。",
        "动态分析使用调试器(如GDB、WinDbg)和模拟器(如QEMU)实时监控程序行为，适合分析复杂逻辑。",
        "静态分析不执行程序代码，通过解析二进制文件结构发现潜在问题，适合大规模自动化扫描。",
        "漏洞利用开发通常包括信息泄露、内存布局操作、控制流劫持和权限提升四个阶段。",
        "现代漏洞缓解技术包括DEP(数据执行保护)、ASLR、CFG(控制流防护)和SafeSEH等。",
        "Shellcode是漏洞利用中执行的机器代码片段，现代攻击常使用多阶段Shellcode绕过防护。"
    ]
    try:
        global embed_model
        kb_index = faiss.IndexFlatL2(embedding_dim)
        vectors = embed_model.encode(default_kb_texts)
        kb_index.add(np.array(vectors))
        kb_mapping = dict(enumerate(default_kb_texts))
        logger.info(f"默认知识库初始化完成，包含 {len(default_kb_texts)} 条知识")
    except Exception as e:
        kb_index = None
        kb_mapping = {}
        logger.error(f"默认知识库初始化失败: {str(e)}")

def load_knowledge_base(kb_path):
    try:
        new_index = faiss.IndexFlatL2(embedding_dim)
        text_chunks = []
        
        # 支持更多文件类型
        supported_extensions = (".txt", ".md", ".pdf", ".docx", ".c", ".cpp", ".h", ".py", ".java")
        
        for root, _, files in os.walk(kb_path):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                            # 预处理内容：移除多余空格和特殊字符
                            content = re.sub(r'\s+', ' ', content).strip()
                            
                            # 根据文件类型优化分块
                            if file.lower().endswith((".c", ".cpp", ".h", ".py", ".java")):
                                # 代码文件按函数分块
                                chunks = re.split(r'\n\s*[{}]\s*\n', content)
                            else:
                                # 文本文件按段落分块
                                chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                            
                            if not chunks:
                                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                            
                            text_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning(f"处理文件 {file_path} 时出错: {str(e)}")
        
        if not text_chunks:
            logger.warning(f"知识库 {kb_path} 中没有找到有效内容")
            return False
        
        # 编码和添加知识块
        vectors = embed_model.encode(text_chunks)
        new_index.add(np.array(vectors))
        new_mapping = dict(enumerate(text_chunks))
        
        with knowledge_lock:
            global kb_index, kb_mapping
            if kb_index:
                kb_index.reset()
            kb_index = new_index
            kb_mapping = new_mapping
        
        logger.info(f"知识库加载成功: {len(text_chunks)} 条知识来自 {kb_path}")
        return True
    except Exception as e:
        logger.error(f"加载知识库失败: {str(e)}")
        return False

def initialize_application():
    global embed_model
    try:
        # 添加模型加载重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                embed_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='model_cache')
                logger.info("嵌入模型加载成功")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"模型加载失败，重试中... ({attempt+1}/{max_retries})")
                    continue
                else:
                    logger.error(f"模型加载失败: {str(e)}")
                    return False
        
        initialize_default_knowledge()
        return True
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        return False

if not initialize_application():
    logger.critical("知识库嵌入模型初始化失败，服务无法启动")
    # 在实际部署中应退出应用
    # import sys
    # sys.exit(1)

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<page>.html')
def serve_static_page(page):
    return send_from_directory('static', f'{page}.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # 优先从请求头获取 API Key 和 Model-Type
        api_key = request.headers.get('Authorization')
        model_type = request.headers.get('Model-Type')
        # 兼容 Bearer token 格式
        if api_key and api_key.lower().startswith('bearer '):
            api_key = api_key[7:].strip()
        # 如果请求头没有，则用全局变量
        if not api_key:
            api_key = global_api_key
        if not model_type:
            model_type = global_model_type
        if not api_key or not model_type:
            return jsonify({"error": "请先配置 API 密钥和模型类型"}), 400

        files = request.files.getlist('files')
        if not files or len(files) < 2:
            return jsonify({"error": "请至少上传两个文件"}), 400

        # 保存上传的文件
        saved_paths = []
        for file in files:
            if file.filename == '':
                continue
            filename = str(uuid.uuid4()) + '_' + file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            saved_paths.append(save_path)
            logger.info(f"文件已保存: {save_path}")

        if len(saved_paths) < 2:
            return jsonify({"error": "有效文件不足两个"}), 400

        # 调用 BinaryComparator 进行三重相似度分析
        compare_results = comparator.compare_files(saved_paths, top_k=1)
        if not compare_results:
            return jsonify({"error": "分析失败，未检测到有效的可比对文件对。"}), 400
        
        file1, file2, combined_similarity = compare_results[0]
        # 取出详细相似度
        sig1 = comparator._get_code_signature(file1)
        sig2 = comparator._get_code_signature(file2)
        
        semantic_similarity = float('nan')
        syntactic_similarity = float('nan')
        lexical_similarity = float('nan')
        
        if sig1 and sig2:
            from sklearn.metrics.pairwise import cosine_similarity
            semantic_similarity = float(cosine_similarity([sig1['features']], [sig2['features']])[0][0])
            lexical_similarity = float(comparator._lexical_similarity(sig1['code'], sig2['code']))
            syntactic_similarity = float(comparator._edit_similarity(sig1['code'], sig2['code']))
        else:
            semantic_similarity = syntactic_similarity = lexical_similarity = 0.0

        # AI 分析摘要
        from binary_analyzer import AIModelAnalyzer
        ai_analyzer = AIModelAnalyzer(api_key)
        
        # 提取部分代码用于分析（避免过长）
        code1 = sig1['code'][:1500] if sig1 else "无法获取代码"
        code2 = sig2['code'][:1500] if sig2 else "无法获取代码"
        
        analysis_result = ai_analyzer.generate_analysis(code1, code2, combined_similarity)

        return jsonify({
            "combined_similarity": round(combined_similarity, 4),
            "semantic_similarity": round(semantic_similarity, 4),
            "syntactic_similarity": round(syntactic_similarity, 4),
            "lexical_similarity": round(lexical_similarity, 4),
            "analysis_result": analysis_result
        })
    except Exception as e:
        logger.exception("文件分析过程中发生错误")
        return jsonify({"error": f"分析出错: {str(e)}"}), 500

@app.route('/vuln_scan', methods=['POST'])
def vuln_scan():
    try:
        # 统一API Key/模型类型
        api_key = request.headers.get('Authorization')
        model_type = request.headers.get('Model-Type')
        if api_key and api_key.lower().startswith('bearer '):
            api_key = api_key[7:].strip()
        if not api_key:
            api_key = global_api_key
        if not model_type:
            model_type = global_model_type or 'deepseek'
        if not api_key or not model_type:
            return jsonify({"error": "请先配置 API 密钥和模型类型"}), 400
        
        file = request.files.get('file')
        if not file or file.filename == '':
            return jsonify({"error": "请上传一个有效文件"}), 400
        
        # 保存上传的文件
        filename = str(uuid.uuid4()) + '_' + file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        logger.info(f"漏洞扫描文件已保存: {save_path}")
        
        analyzer = VulnerabilityAnalyzer(model_type=model_type, api_key=api_key)
        result = analyzer.analyze(save_path)
        
        if result.get('status') != 'success':
            return jsonify({"error": result.get('message', '分析失败')}), 500
        
        # 风险分布数据
        features = result['features']
        radar_data = [
            features.get('dangerous_calls', 0),
            features.get('memory_operations', {}).get('buffer_access', 0),
            features.get('cfg_complexity', 0),
            len(features.get('api_sequence', []))
        ]
        
        return jsonify({
            "radar_data": radar_data,
            "ai_report": result['report'],
            "features": features
        })
    except Exception as e:
        logger.exception("漏洞扫描过程中发生错误")
        return jsonify({"error": f"分析出错: {str(e)}"}), 500

@app.route('/config', methods=['POST'])
def set_config():
    try:
        data = request.json
        if not data or 'apiKey' not in data or 'modelType' not in data:
            return jsonify({"error": "缺少必要参数"}), 400
        
        global global_api_key, global_model_type
        global_api_key = data['apiKey'].strip()
        global_model_type = data['modelType'].strip()
        
        logger.info(f"配置已更新: API Key={global_api_key[:4]}..., Model={global_model_type}")
        return jsonify({"success": True, "message": "配置已保存"})
    except Exception as e:
        logger.error(f"配置更新失败: {str(e)}")
        return jsonify({"error": "配置保存失败"}), 500

@app.route('/api/get-ai-report', methods=['POST'])
def get_ai_report():
    return jsonify({"error": "该接口尚未启用"}), 501

@app.route("/start_rag", methods=["POST"])
def start_rag_service():
    try:
        global rag_service_active, knowledge_base_path, global_api_key
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "无效请求"}), 400
        
        kb_path = data.get("kb_path", "")
        api_key = data.get("api_key", "")
        
        # 统一API密钥配置
        if api_key:
            global global_api_key
            global_api_key = api_key.strip()
            logger.info("API Key已通过RAG服务设置")
        
        if kb_path and not os.path.isdir(kb_path):
            return jsonify({"status": "error", "message": "知识库路径不存在"}), 400
        
        try:
            if kb_path:
                with knowledge_lock:
                    success = load_knowledge_base(kb_path)
                    if success:
                        knowledge_base_path = kb_path
                        rag_service_active = True
                        logger.info(f"RAG服务已启动，知识库: {kb_path}")
                        return jsonify({"status": "success", "message": "RAG服务已启动"})
                    else:
                        return jsonify({"status": "error", "message": "知识库加载失败"}), 500
            else:
                rag_service_active = True
                logger.info("RAG服务已启动，使用默认知识库")
                return jsonify({"status": "success", "message": "使用默认知识库启动RAG服务"})
        except Exception as e:
            logger.error(f"启动RAG服务失败: {str(e)}")
            return jsonify({"status": "error", "message": f"内部错误: {str(e)}"}), 500
    except Exception as e:
        logger.exception("启动RAG服务时发生异常")
        return jsonify({"status": "error", "message": "服务器错误"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        global rag_service_active, global_api_key, global_model_type
        
        if not rag_service_active:
            return jsonify({"error": "RAG服务未启动，请先启动服务"}), 503
        
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "问题不能为空"}), 400
        
        user_question = data["question"].strip()
        if not user_question:
            return jsonify({"error": "问题不能为空"}), 400
        
        logger.info(f"收到问题: {user_question}")
        
        try:
            with knowledge_lock:
                if not kb_index or not kb_mapping:
                    return jsonify({"error": "知识库未初始化"}), 500
                
                query_vec = embed_model.encode([user_question])
                D, I = kb_index.search(np.array(query_vec), k=3)
                
                retrieved = []
                for idx in I[0]:
                    if idx in kb_mapping and idx >= 0:  # 确保索引有效
                        retrieved.append(kb_mapping[idx])
            
            context = "\n\n".join(retrieved)
            
            # 优化系统提示词，使回答更人性化
            system_prompt = (
                "你是一个精通二进制分析和漏洞研究的AI助手，具有专业知识和友好态度。\n"
                "请根据以下知识库内容，用清晰、简洁且人性化的方式回答用户问题。\n"
                "回答时应：\n"
                "1. 使用自然的对话语气，避免过于技术化的术语堆砌\n"
                "2. 对复杂概念进行通俗易懂的解释\n"
                "3. 在适当处加入实际示例帮助理解\n"
                "4. 当知识库内容不足时，可补充你的专业知识但需注明\n"
                "5. 保持回答结构清晰，必要时分段说明\n\n"
                "知识库内容如下：\n"
                "---------------\n"
                f"{context}\n"
                "---------------\n\n"
                f"用户问题: {user_question}"
            )
            
            from model_handler import ModelHandler
            handler = ModelHandler(api_key=global_api_key, default_model=global_model_type or "deepseek")
            
            # 添加人性化参数
            answer = handler.generate_analysis(
                system_prompt,
                temperature=0.7,  # 增加创造性
                max_tokens=800,   # 允许更长回答
                top_p=0.9,        # 增加多样性
                presence_penalty=0.5,  # 减少重复
                frequency_penalty=0.5  # 减少重复
            )
            
            logger.info(f"问题已回答，长度: {len(answer)} 字符")
            return jsonify({"answer": answer})
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            return jsonify({"error": f"处理问题时出错: {str(e)}"}), 500
    except Exception as e:
        logger.exception("处理问题请求时发生异常")
        return jsonify({"error": "服务器错误"}), 500

@app.route("/status", methods=["GET"])
def service_status():
    try:
        with knowledge_lock:
            return jsonify({
                "active": rag_service_active,
                "knowledge_base": knowledge_base_path or "默认知识库",
                "documents_count": len(kb_mapping),
                "model": global_model_type or "未设置"
            })
    except Exception:
        return jsonify({"status": "error"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "model_loaded": embed_model is not None,
            "index_ready": kb_index is not None,
            "rag_active": rag_service_active,
            "memory_usage": f"{os.sys.getsizeof(globals()) // 1024} KB"
        })
    except Exception:
        return jsonify({"status": "unhealthy"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
