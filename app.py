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

from binary_analyzer import BinaryComparator
from plagiarism_detector import run_plagiarism_check
from to_be_used.vulnerability_detector import VulnerabilityAnalyzer

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.secret_key = ''
comparator = BinaryComparator("D:\\Program\\CodeT5\\codet5")

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

atexit.register(cleanup_resources)

def initialize_default_knowledge():
    global kb_index, kb_mapping
    default_kb_texts = [
        "缓冲区溢出是一种内存操作漏洞，攻击者可以覆盖返回地址。",
        "格式化字符串漏洞可以导致任意内存读取。",
        "整数溢出发生在算术运算结果超出变量类型范围时。",
        "堆溢出漏洞利用堆内存分配机制进行攻击。",
        "释放后使用(UAF)漏洞发生在程序继续使用已释放的内存时。"
    ]
    try:
        global embed_model
        kb_index = faiss.IndexFlatL2(embedding_dim)
        vectors = embed_model.encode(default_kb_texts)
        kb_index.add(np.array(vectors))
        kb_mapping = dict(enumerate(default_kb_texts))
    except Exception as e:
        kb_index = None
        kb_mapping = {}

def load_knowledge_base(kb_path):
    try:
        new_index = faiss.IndexFlatL2(embedding_dim)
        text_chunks = []
        for root, _, files in os.walk(kb_path):
            for file in files:
                if file.lower().endswith((".txt", ".md", ".pdf", ".docx")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                            if not chunks:
                                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                            text_chunks.extend(chunks)
                    except Exception:
                        pass
        if not text_chunks:
            return False
        vectors = embed_model.encode(text_chunks)
        new_index.add(np.array(vectors))
        new_mapping = dict(enumerate(text_chunks))
        with knowledge_lock:
            global kb_index, kb_mapping
            if kb_index:
                kb_index.reset()
            kb_index = new_index
            kb_mapping = new_mapping
        return True
    except Exception:
        return False

def initialize_application():
    global embed_model
    try:
        try:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='model_cache')
        except:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        initialize_default_knowledge()
        return True
    except Exception:
        return False

if not initialize_application():
    print("知识库嵌入模型初始化失败，服务无法启动")

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<page>.html')
def serve_static_page(page):
    return send_from_directory('static', f'{page}.html')

@app.route('/upload', methods=['POST'])
def upload_files():
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
        filename = str(uuid.uuid4()) + '_' + file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        saved_paths.append(save_path)

    try:
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
        analysis_result = ai_analyzer.generate_analysis(sig1['code'][:1000], sig2['code'][:1000], combined_similarity)

        return jsonify({
            "combined_similarity": round(combined_similarity, 4),
            "semantic_similarity": round(semantic_similarity, 4),
            "syntactic_similarity": round(syntactic_similarity, 4),
            "lexical_similarity": round(lexical_similarity, 4),
            "analysis_result": analysis_result
        })
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": f"分析出错: {str(e)}"}), 500

@app.route('/vuln_scan', methods=['POST'])
def vuln_scan():
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
    if not file:
        return jsonify({"error": "请上传一个文件"}), 400
    # 保存上传的文件
    filename = str(uuid.uuid4()) + '_' + file.filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    try:
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
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": f"分析出错: {str(e)}"}), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    global global_api_key, global_model_type
    data = request.get_json()
    if not data or 'apiKey' not in data or 'modelType' not in data:
        return jsonify({"error": "缺少参数"}), 400
    global_api_key = data['apiKey'].strip()
    global_model_type = data['modelType'].strip()
    return jsonify({"success": True, "message": "配置已保存"})

@app.route('/api/get-ai-report', methods=['POST'])
def get_ai_report():
    return jsonify({"error": "该接口尚未启用"}), 501

@app.route("/start_rag", methods=["POST"])
def start_rag_service():
    global rag_service_active, knowledge_base_path, global_api_key
    data = request.json
    kb_path = data.get("kb_path", "")
    api_key = data.get("api_key", "")
    # 统一API密钥配置
    if api_key:
        global global_api_key
        global_api_key = api_key.strip()
    if kb_path and not os.path.isdir(kb_path):
        return jsonify({"status": "error", "message": "知识库路径不存在"}), 400
    try:
        if kb_path:
            with knowledge_lock:
                success = load_knowledge_base(kb_path)
                if success:
                    knowledge_base_path = kb_path
                    rag_service_active = True
                    return jsonify({"status": "success", "message": "RAG服务已启动"})
                else:
                    return jsonify({"status": "error", "message": "知识库加载失败"}), 500
        else:
            rag_service_active = True
            return jsonify({"status": "success", "message": "使用默认知识库启动RAG服务"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"内部错误: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    global rag_service_active, global_api_key, global_model_type
    if not rag_service_active:
        return jsonify({"error": "RAG服务未启动，请先启动服务"}), 503
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"error": "问题不能为空"}), 400
    try:
        with knowledge_lock:
            if not kb_index or not kb_mapping:
                return jsonify({"error": "知识库未初始化"}), 500
            query_vec = embed_model.encode([user_question])
            D, I = kb_index.search(np.array(query_vec), k=3)
            retrieved = []
            for idx in I[0]:
                if idx in kb_mapping:
                    retrieved.append(kb_mapping[idx])
        context = "\n\n".join(retrieved)
        system_prompt = (
            "你是一个精通二进制分析和漏洞研究的AI助手，回答必须基于提供的知识库内容。\n"
            "知识库内容如下：\n"
            "---------------\n"
            f"{context}\n"
            "---------------\n"
            "请严格根据上述知识库内容回答用户问题。如果知识库中没有相关信息，请说明无法回答。"
            "若用户要求的答案超出知识库范围，请参考网络资料进行回答，并注明出处。"
        )
        from model_handler import ModelHandler
        handler = ModelHandler(api_key=global_api_key, default_model=global_model_type or "deepseek")
        answer = handler.generate_analysis(system_prompt)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"处理问题时出错: {str(e)}"}), 500

@app.route("/status", methods=["GET"])
def service_status():
    with knowledge_lock:
        return jsonify({
            "active": rag_service_active,
            "knowledge_base": knowledge_base_path or "默认知识库",
            "documents_count": len(kb_mapping)
        })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": embed_model is not None,
        "index_ready": kb_index is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
