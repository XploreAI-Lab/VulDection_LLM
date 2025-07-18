import os
import hashlib
import logging
import numpy as np
import torch
import heapq
import pefile
# import tlsh
from elftools.elf.elffile import ELFFile
from capstone import *
from collections import OrderedDict
from rapidfuzz.distance import Levenshtein
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from model_handler import ModelHandler
from PyQt5.QtCore import QMutex, QMutexLocker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DisassemblyEngine:
    """增强型反汇编引擎，支持精确架构检测"""

    MAX_CODE_SIZE = 10 * 1024 * 1024  # 最大代码段大小（10MB）
    ARCH_MAP = {
        'x86': (CS_ARCH_X86, CS_MODE_32),
        'x64': (CS_ARCH_X86, CS_MODE_64),
        'arm': (CS_ARCH_ARM, CS_MODE_ARM),
        'arm64': (CS_ARCH_ARM64, CS_MODE_ARM)
    }

    def __init__(self):
        self.md = None
        self.current_arch = None

    def _detect_architecture(self, file_path):
        """精确架构检测方法"""
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic.startswith(b'MZ'):
                    try:
                        pe = pefile.PE(file_path)
                        machine = pe.FILE_HEADER.Machine
                        return 'x64' if machine == 0x8664 else 'x86'
                    except Exception as pe_err:
                        logging.error(f"PE解析失败: {str(pe_err)}")
                elif magic == b'\x7fELF':
                    try:
                        with open(file_path, 'rb') as elf_file:
                            elf = ELFFile(elf_file)
                            machine = elf.header['e_machine']
                            return {0x3E: 'x64', 0x03: 'x86', 0x28: 'arm'}.get(machine, 'unknown')
                    except Exception as elf_err:
                        logging.error(f"ELF解析失败: {str(elf_err)}")
                return 'unknown'
        except Exception as e:
            logging.error(f"架构检测失败: {str(e)}")
            return 'unknown'

    def _extract_code_section(self, file_path):
        """改进的代码段提取方法"""
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                f.seek(0)
                if magic.startswith(b'MZ'):
                    pe = pefile.PE(data=f.read())
                    return next((section.get_data() for section in pe.sections if b".text" in section.Name), None)
                elif magic == b'\x7fELF':
                    elf = ELFFile(f)
                    return next((segment.data() for segment in elf.iter_segments() if
                                 segment['p_type'] == 'PT_LOAD' and segment['p_flags'] & 0x1), None)
        except Exception as e:
            logging.error(f"代码段提取失败: {str(e)}")
            return None

    def disassemble(self, file_path):
        """安全反汇编方法"""
        code_data = self._extract_code_section(file_path)
        if not code_data:
            return None

        arch_type = self._detect_architecture(file_path)
        if arch_type not in self.ARCH_MAP:
            return None

        if len(code_data) > self.MAX_CODE_SIZE:
            code_data = code_data[:self.MAX_CODE_SIZE]
            logging.warning(f"代码段过长，截断至{self.MAX_CODE_SIZE}字节")

        if arch_type != self.current_arch:
            arch, mode = self.ARCH_MAP[arch_type]
            self.md = Cs(arch, mode)
            self.current_arch = arch_type

        try:
            disasm = []
            for ins in self.md.disasm(code_data, 0x1000):
                if not ins.mnemonic.lower().startswith(('db', 'dw', 'dd')):
                    disasm.append(f"{ins.mnemonic} {ins.op_str}")
            return '\n'.join(disasm)
        except CsError as e:
            logging.error(f"反汇编错误: {str(e)}")
            return None


class SemanticAnalyzer:
    """修正特征提取的语义分析模块"""

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self._load_model(model_path)
        self.pca = PCA(n_components=128)
        self.pca_fitted = False

    def _load_model(self, model_path):
        """安全的本地模型加载"""
        required_files = ['config.json', 'pytorch_model.bin']
        for fname in required_files:
            if not os.path.exists(os.path.join(model_path, fname)):
                raise FileNotFoundError(f"缺失模型文件: {fname}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            return tokenizer, model
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            raise

    def get_features(self, code_list):
        """修正后的特征提取方法"""
        features = []
        for code in code_list:
            try:
                inputs = self.tokenizer(
                    code,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    encoder_output = self.model.encoder(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask
                    )

                feature = torch.mean(encoder_output.last_hidden_state, dim=1).cpu().numpy().flatten()
                features.append(feature)
            except Exception as e:
                logging.error(f"特征提取失败: {str(e)}")
                features.append(np.zeros(self.model.config.d_model))

        features = np.array(features)
        if len(features) > 1 and not self.pca_fitted:
            self.pca.fit(features)
            self.pca_fitted = True
        return self.pca.transform(features) if self.pca_fitted else features


class BinaryComparator:
    """支持架构感知的三重视似度计算"""
    MAX_CACHE_SIZE = 10000  # 最大缓存大小（10000条）
    def __init__(self, model_path):
        self.disassembler = DisassemblyEngine()
        self.analyzer = SemanticAnalyzer(model_path)
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = logging.getLogger("BinaryComparator")
        self.cache_mutex = QMutex()

    def get_cache_status(self):
        """获取缓存状态"""
        return {
            'total_entries': len(self.cache),
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0

        }
    def _preprocess_code(self, code):
        """增强的代码预处理"""
        cleaned = []
        for line in code.split('\n'):
            clean_line = line.split(';')[0].split(':')[-1].strip().lower().replace('%', '')
            if clean_line:
                cleaned.append(clean_line)
        return '\n'.join(cleaned)

    def _get_code_signature(self, file_path):
        """带哈希校验的缓存加载"""
        # 生成包含文件内容的缓存键
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        cache_key = f"{file_path}_{file_hash}"

        with QMutexLocker(self.cache_mutex):
            if cache_key in self.cache:
                self.cache_hits += 1
                self.cache.move_to_end(cache_key)
                return self.cache[cache_key]

            self.cache_misses += 1
            try:
                disasm = self.disassembler.disassemble(file_path)
                if not disasm:
                    self.logger.warning(f"反汇编失败：{file_path}")
                    return None

                clean_code = self._preprocess_code(disasm)
                features = self.analyzer.get_features([clean_code])[0]

                if len(self.cache) >= self.MAX_CACHE_SIZE:
                    self.cache.popitem(last=False)

                # 添加新条目至缓存
                self.cache[cache_key] = {
                    'code': clean_code,
                    'features': features,
                    'arch': self.disassembler.current_arch
                }
                return self.cache[cache_key]
            except Exception as e:
                self.logger.error(f"缓存填充失败: {str(e)}")
                return None

    def compare_files(self, file_paths, top_k=10):
        """架构感知的三重比对方法"""
        signatures = {}
        for path in file_paths:
            if sig := self._get_code_signature(path):
                signatures[path] = sig

        results = []
        paths = list(signatures.keys())
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                file1, file2 = paths[i], paths[j]
                sig1, sig2 = signatures[file1], signatures[file2]

                if sig1['arch'] != sig2['arch']:
                    continue

                # 语义相似度
                cos_sim = cosine_similarity([sig1['features']], [sig2['features']])[0][0]
                # 词汇相似度
                lexical_sim = self._lexical_similarity(sig1['code'], sig2['code'])
                # 语法相似度
                edit_sim = self._edit_similarity(sig1['code'], sig2['code'])

                combined = 0.5 * cos_sim + 0.3 * lexical_sim + 0.2 * edit_sim
                results.append((file1, file2, combined))

        return heapq.nlargest(top_k, results, key=lambda x: x[2])

    def _lexical_similarity(self, code1, code2):
        mnemonics1 = {line.split()[0] for line in code1.split('\n') if line.strip()}
        mnemonics2 = {line.split()[0] for line in code2.split('\n') if line.strip()}
        if not mnemonics1 and not mnemonics2:
            return 1.0
        intersection = len(mnemonics1 & mnemonics2)
        union = len(mnemonics1 | mnemonics2)
        return intersection / union if union else 0.0

    def _edit_similarity(self, code1, code2):
        tokens1 = code1.split()
        tokens2 = code2.split()
        max_len = max(len(tokens1), len(tokens2))
        edit_dist = Levenshtein.distance(' '.join(tokens1), ' '.join(tokens2))
        return 1 - (edit_dist / max_len) if max_len else 0.0



    def _memory_usage_over_threshold(self, threshold=0.8):
        """动态内存阈值检测"""
        try:
            import psutil
            total = psutil.virtual_memory().total
            used = psutil.virtual_memory().used
            return (used / total) > threshold
        except:
            return False

    BLOCK_SIZE = 4096  # 哈希计算块大小（字节）
    MIN_SIMILARITY = 0.3  # 相似度有效阈值

    def _streaming_hash(self, file_obj):
        """使用多级分块哈希算法"""
        import hashlib
        content_hashes = []
        while True:
            data = file_obj.read(self.BLOCK_SIZE)
            if not data:
                break
            # 双重哈希增强稳定性
            block_hash = hashlib.md5(data).hexdigest()  # 快速哈希
            content_hash = hashlib.sha256(data).hexdigest()  # 精确哈希
            content_hashes.append(f"{block_hash}:{content_hash}")
        return content_hashes

    def _calc_similarity(self, hash_list1, hash_list2):
        """带权重优化的相似度计算"""
        # 空文件处理
        if not hash_list1 and not hash_list2:
            return 1.0
        if not hash_list1 or not hash_list2:
            return 0.0

        # 动态权重分配
        max_len = max(len(hash_list1), len(hash_list2))
        weights = [0.4 * (i/max_len) + 0.6 for i in range(max_len)]  # 后半部分权重更高
        
        # 对齐处理
        aligned_pairs = []
        for i in range(min(len(hash_list1), len(hash_list2))):
            h1 = hash_list1[i]
            h2 = hash_list2[i]
            # MD5部分对比
            if h1.split(':')[0] == h2.split(':')[0]:
                aligned_pairs.append(1.0)
            else:
                # SHA256精确对比
                aligned_pairs.append(1.0 if h1 == h2 else 0.0)
        
        # 加权计算
        weighted_sum = sum(w * s for w, s in zip(weights[:len(aligned_pairs)], aligned_pairs))
        total_weight = sum(weights[:len(aligned_pairs)])
        
        # 长度差异惩罚
        length_penalty = 1 - abs(len(hash_list1) - len(hash_list2)) / max_len
        
        final_score = (weighted_sum / total_weight) * length_penalty
        return max(min(final_score, 1.0), 0.0)

    def compare_pair(self, file1, file2):
        """增强的哈希对比方法"""
        try:
            with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                # 使用内存优化流式处理
                def generate_hashes(f):
                    hasher = []
                    while True:
                        data = f.read(self.BLOCK_SIZE)
                        if not data:
                            break
                        block_hash = hashlib.md5(data).hexdigest()
                        hasher.append(block_hash)
                    return hasher
                
                hash_list1 = generate_hashes(f1)
                hash_list2 = generate_hashes(f2)
                
            similarity = self._calc_similarity(hash_list1, hash_list2)
            return similarity if similarity >= self.MIN_SIMILARITY else 0.0
            
        except Exception as e:
            logging.error(f"传统哈希对比失败: {str(e)}")
            return 0.0


    # 在调用get_functions前增加路径验证
    def update_ranking(self, files):
        for path in files:
            # 添加路径有效性检查
            if not os.path.isfile(path):
                print(f"无效文件路径: {path}")
                continue

            # 添加二进制文件验证
            with open(path, 'rb') as f:
                header = f.read(4)
                if not header.startswith(b'\x7fELF') and not header.startswith(b'MZ'):
                    print(f"非可执行文件: {path}")
                    continue

            funcs = self.comparator.get_functions(path)  # 现在传入的是真实路径

    def get_functions(self, file_path):
        """获取文件函数列表（整合FunctionAnalyzer功能）"""
        try:
            # 初始化函数分析器
            func_analyzer = FunctionAnalyzer(self.disassembler)
            # 获取反汇编代码
            disasm = self.disassembler.disassemble(file_path)
            return func_analyzer.extract_functions(disasm) if disasm else []
        except Exception as e:
            logging.error(f"函数提取失败: {str(e)}")
            return []


class AIModelAnalyzer:
    """保留原有大模型分析模块"""

    def __init__(self, api_key: str):
        self.model_handler = ModelHandler(api_key=api_key, default_model='deepseek')
    # 统一调用

    def generate_analysis(self, code1: str, code2: str, score: float) -> str:
        prompt = f"""二进制代码分析任务：

                ## 基础信息
                - 相似度得分：{score:.2f}
                - 代码片段1特征：{code1[:500]}
                - 代码片段2特征：{code2[:500]}

                ## 分析要求
                - 功能相似性分析
                - 安全风险对比
                - 混淆迹象识别
                - 架构差异说明"""

        return self.model_handler.generate_analysis(prompt)


# 在binary_analyzer.py中新增函数分析类
class FunctionAnalyzer:
    def __init__(self, disassembler):
        self.disassembler = disassembler

    def extract_functions(self, file_path):
        """使用控制流分析识别函数边界"""
        disasm = self.disassembler.disassemble(file_path)
        # 基于跳转目标划分函数（简化实现）
        functions = []
        current_func = []
        for line in disasm.split('\n'):
            if 'call' in line or 'jmp' in line:
                if current_func:
                    functions.append('\n'.join(current_func))
                    current_func = []
            else:
                current_func.append(line)
        return functions


