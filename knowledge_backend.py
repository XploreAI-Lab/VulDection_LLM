import os
import logging
from threading import Lock
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import atexit

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_service.log')
    ]
)
logger = logging.getLogger("RAG_Service")

# 创建应用实例
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# 全局变量
embed_model = None
embedding_dim = 384  # MiniLM 默认维度
kb_index = None
kb_mapping = {}
knowledge_base_path = ""
rag_service_active = False
api_key = ""
knowledge_lock = Lock()

# 清理函数
def cleanup_resources():
    global kb_index, kb_mapping
    if kb_index:
        logger.info("清理FAISS索引资源")
        kb_index.reset()
        kb_index = None
        kb_mapping = {}

# 注册退出清理函数
atexit.register(cleanup_resources)

# 初始化默认知识库
def initialize_default_knowledge():
    global kb_index, kb_mapping
    default_kb_texts = [
        "缓冲区溢出是一种内存操作漏洞，攻击者可以覆盖返回地址。",
        "格式化字符串漏洞可以导致任意内存读取。",
        "整数溢出发生在算术运算结果超出变量类型范围时。",
        "堆溢出漏洞利用堆内存分配机制进行攻击。",
        "释放后使用(UAF)漏洞发生在程序继续使用已释放的内存时。"
    ]
    
    # 初始化索引
    try:
        kb_index = faiss.IndexFlatL2(embedding_dim)
        vectors = embed_model.encode(default_kb_texts)
        kb_index.add(np.array(vectors))
        kb_mapping = dict(enumerate(default_kb_texts))
        logger.info("默认知识库已初始化")
    except Exception as e:
        logger.error(f"初始化默认知识库失败: {str(e)}")
        kb_index = None
        kb_mapping = {}

# 加载知识库
def load_knowledge_base(kb_path):
    try:
        # 创建新的索引和映射
        new_index = faiss.IndexFlatL2(embedding_dim)
        new_mapping = {}
        
        # 处理所有文本文件
        text_chunks = []
        for root, _, files in os.walk(kb_path):
            for file in files:
                if file.lower().endswith((".txt", ".md", ".pdf", ".docx")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # 更智能的分块 (按段落分块)
                            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                            if not chunks:  # 如果没有空行分隔，使用固定大小分块
                                chunks = [content[i:i+500] for i in range(0, len(content), 500)]
                            text_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"读取文件 {file_path} 失败: {str(e)}")
        
        if not text_chunks:
            logger.warning("知识库目录中没有找到有效的文本文件")
            return False
        
        # 编码并添加到索引
        vectors = embed_model.encode(text_chunks)
        new_index.add(np.array(vectors))
        
        # 创建映射
        new_mapping = dict(enumerate(text_chunks))
        logger.info(f"知识库加载成功, 共 {len(text_chunks)} 个片段")
        
        # 更新全局索引和映射
        with knowledge_lock:
            global kb_index, kb_mapping
            if kb_index:
                kb_index.reset()
            kb_index = new_index
            kb_mapping = new_mapping
        
        return True
    except Exception as e:
        logger.error(f"加载知识库失败: {str(e)}")
        return False

# 应用初始化函数
def initialize_application():
    global embed_model
    logger.info("正在初始化嵌入模型...")
    try:
        # 尝试从本地缓存加载模型
        try:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='model_cache')
        except:
            # 如果本地缓存失败，从网络加载
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 初始化默认知识库
        initialize_default_knowledge()
        logger.info("应用初始化完成")
        return True
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        return False

# 在应用启动时初始化
if not initialize_application():
    logger.critical("应用初始化失败，服务无法启动")
    # 在实际部署中，这里应该退出应用
    # import sys
    # sys.exit(1)

# 服务前端页面
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'knowledge.html')

# 启动RAG服务
@app.route("/start_rag", methods=["POST"])
def start_rag_service():
    global rag_service_active, knowledge_base_path, api_key
    
    data = request.json
    kb_path = data.get("kb_path", "")
    api_key = data.get("api_key", "")
    
    # 验证API密钥
    if not api_key:
        return jsonify({"status": "error", "message": "API密钥不能为空"}), 400
    
    # 验证知识库路径
    if kb_path and not os.path.isdir(kb_path):
        return jsonify({"status": "error", "message": "知识库路径不存在"}), 400
    
    try:
        # 如果有自定义知识库路径，加载它
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
            # 使用默认知识库
            rag_service_active = True
            return jsonify({"status": "success", "message": "使用默认知识库启动RAG服务"})
    except Exception as e:
        logger.error(f"启动RAG服务失败: {str(e)}")
        return jsonify({"status": "error", "message": f"内部错误: {str(e)}"}), 500

# 问答接口
@app.route("/ask", methods=["POST"])
def ask_question():
    global rag_service_active
    
    if not rag_service_active:
        return jsonify({"error": "RAG服务未启动，请先启动服务"}), 503
    
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"error": "问题不能为空"}), 400
    
    try:
        # 获取相关内容
        with knowledge_lock:
            if not kb_index or not kb_mapping:
                return jsonify({"error": "知识库未初始化"}), 500
                
            query_vec = embed_model.encode([user_question])
            D, I = kb_index.search(np.array(query_vec), k=3)
            
            # 检索相关文本
            retrieved = []
            for idx in I[0]:
                if idx in kb_mapping:
                    retrieved.append(kb_mapping[idx])
        
        # 组装提示词
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
        
        # 使用DeepSeek生成回答
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=2000,
            temperature=0.3,
            stream=False
        )
        
        answer = response.choices[0].message.content
        
        # 记录问答日志
        logger.info(f"问题: {user_question}\n回答: {answer[:200]}...")
        
        return jsonify({"answer": answer})
    
    except Exception as e:
        logger.error(f"问答处理失败: {str(e)}")
        return jsonify({"error": f"处理问题时出错: {str(e)}"}), 500

# 服务状态检查
@app.route("/status", methods=["GET"])
def service_status():
    with knowledge_lock:
        return jsonify({
            "active": rag_service_active,
            "knowledge_base": knowledge_base_path or "默认知识库",
            "documents_count": len(kb_mapping)
        })

# 健康检查端点
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": embed_model is not None,
        "index_ready": kb_index is not None
    })

# 提供静态文件服务
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        cleanup_resources()
