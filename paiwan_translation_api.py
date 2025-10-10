import re
import json
from typing import List, Dict, Any
from test import QdrantClientWrapper, create_ollama_embeddings, create_mock_embeddings
from fuzzywuzzy import fuzz
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import uvicorn

class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    original_text: str
    translation: str
    success: bool
    direction: str
    
class TranslationResponseonlytext(BaseModel):
    translation: str

class PaiwanToChineseTranslator:
    def __init__(self, collection_name: str = "paiwan_enhanced"):
        self.qdrant = QdrantClientWrapper()
        self.collection_name = collection_name
        self.embedding_model = "nomic-embed-text"
        self.ollama_host = "http://localhost:11434"
        
        # 建立詞彙字典
        self.vocab_dict = self._build_vocabulary_dict()
        
    def _build_vocabulary_dict(self) -> Dict[str, Dict]:
        """建立排灣語-中文詞彙對照字典"""
        vocab_dict = {'paiwan_to_chinese': {}}
        
        # 從JSON文件中讀取詞彙
        try:
            with open('data/unique_data.json', 'r', encoding='utf-8') as f:
                word_list = json.load(f)
            
            # 處理每個詞彙對
            for entry in word_list:
                paiwan_word = entry.get("paiwan", "").strip()
                chinese_meaning = entry.get("chinese", "").strip()
                
                # 跳過空白或無效的條目
                if not paiwan_word or not chinese_meaning:
                    continue
                
                # 跳過標記為 [虛] 的詞
                if chinese_meaning == '[虛]' or chinese_meaning == '[虛':
                    continue
                
                # 添加到字典
                if paiwan_word not in vocab_dict['paiwan_to_chinese']:
                    vocab_dict['paiwan_to_chinese'][paiwan_word] = []
                
                if chinese_meaning not in vocab_dict['paiwan_to_chinese'][paiwan_word]:
                    vocab_dict['paiwan_to_chinese'][paiwan_word].append(chinese_meaning)
            
            # 輸出統計信息以便調試
            print(f"建立了 {len(vocab_dict['paiwan_to_chinese'])} 個排灣語詞彙的對照")
        
        except Exception as e:
            print(f"讀取詞彙時出錯: {str(e)}")
        
        return vocab_dict
    
    def translate(self, paiwan_text: str) -> str:
        """翻譯排灣語為中文"""
        all_translations = []
        
        # 精確匹配
        if paiwan_text in self.vocab_dict['paiwan_to_chinese']:
            all_translations.extend(self.vocab_dict['paiwan_to_chinese'][paiwan_text])
        
        # 模糊匹配
        best_score = 0
        for word, translations in self.vocab_dict['paiwan_to_chinese'].items():
            similarity = fuzz.ratio(paiwan_text.lower(), word.lower())
            if similarity > 80 and word != paiwan_text:
                if similarity > best_score:
                    best_score = similarity
                    all_translations.extend(translations)
        
        # 如果詞彙匹配沒有結果，進行語境搜尋
        if not all_translations:
            try:
                query_vector = create_ollama_embeddings(paiwan_text, self.embedding_model, self.ollama_host)
            except:
                query_vector = create_mock_embeddings(paiwan_text)
            
            results = self.qdrant.search_similar(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=3,
                score_threshold=0.6
            )
            
            if results:
                translation = self._extract_translation_from_content(paiwan_text, results[0]['content'])
                if translation:
                    all_translations.append(translation)
        
        # 去除重複並保持順序
        unique_translations = []
        seen = set()
        for trans in all_translations:
            trans_clean = trans.strip()
            if trans_clean and trans_clean not in seen:
                unique_translations.append(trans_clean)
                seen.add(trans_clean)
        
        if unique_translations:
            return ', '.join(unique_translations)
        else:
            return paiwan_text  # 查不到就回傳原本內容
    
    def _extract_translation_from_content(self, query: str, content: str) -> str:
        """從內容中提取翻譯"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith('排灣語：') and query in line:
                if i - 1 >= 0 and lines[i - 1].startswith('中文：'):
                    return lines[i - 1][3:].strip()
        
        return ""

class ChineseToPaiwanTranslator:
    def __init__(self, collection_name: str = "paiwan_enhanced"):
        self.qdrant = QdrantClientWrapper()
        self.collection_name = collection_name
        self.embedding_model = "nomic-embed-text"
        self.ollama_host = "http://localhost:11434"
        
        # 建立詞彙字典
        self.vocab_dict = self._build_vocabulary_dict()
        
    def _build_vocabulary_dict(self) -> Dict[str, Dict]:
        """建立中文-排灣語詞彙對照字典"""
        vocab_dict = {'chinese_to_paiwan': {}}
        
        # 從JSON文件中讀取詞彙
        try:
            with open('data/paiwan_words.json', 'r', encoding='utf-8') as f:
                word_list = json.load(f)
            
            # 處理每個詞彙對
            for entry in word_list:
                paiwan_word = entry.get("paiwan", "").strip()
                chinese_meaning = entry.get("chinese", "").strip()
                
                # 跳過空白或無效的條目
                if not paiwan_word or not chinese_meaning:
                    continue
                
                # 跳過標記為 [虛] 的詞
                if chinese_meaning == '[虛]' or chinese_meaning == '[虛':
                    continue
                
                # 添加到字典
                if chinese_meaning not in vocab_dict['chinese_to_paiwan']:
                    vocab_dict['chinese_to_paiwan'][chinese_meaning] = []
                
                if paiwan_word not in vocab_dict['chinese_to_paiwan'][chinese_meaning]:
                    vocab_dict['chinese_to_paiwan'][chinese_meaning].append(paiwan_word)
            
            # 輸出統計信息以便調試
            print(f"建立了 {len(vocab_dict['chinese_to_paiwan'])} 個中文詞彙的對照")
        
        except Exception as e:
            print(f"讀取詞彙時出錯: {str(e)}")
        
        return vocab_dict
    
    def translate(self, chinese_text: str) -> str:
        """翻譯中文為排灣語"""
        all_translations = []
        
        # 精確匹配
        if chinese_text in self.vocab_dict['chinese_to_paiwan']:
            all_translations.extend(self.vocab_dict['chinese_to_paiwan'][chinese_text])
        
        # 模糊匹配
        best_score = 0
        for word, translations in self.vocab_dict['chinese_to_paiwan'].items():
            similarity = fuzz.ratio(chinese_text.lower(), word.lower())
            if similarity > 80 and word != chinese_text:
                if similarity > best_score:
                    best_score = similarity
                    all_translations.extend(translations)
        
        # 如果詞彙匹配沒有結果，進行語境搜尋
        if not all_translations:
            try:
                query_vector = create_ollama_embeddings(chinese_text, self.embedding_model, self.ollama_host)
            except:
                query_vector = create_mock_embeddings(chinese_text)
            
            results = self.qdrant.search_similar(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=3,
                score_threshold=0.6
            )
            
            if results:
                translation = self._extract_translation_from_content(chinese_text, results[0]['content'])
                if translation:
                    all_translations.append(translation)
        
        # 去除重複並保持順序
        unique_translations = []
        seen = set()
        for trans in all_translations:
            trans_clean = trans.strip()
            if trans_clean and trans_clean not in seen:
                unique_translations.append(trans_clean)
                seen.add(trans_clean)
        
        if unique_translations:
            return ', '.join(unique_translations)
        else:
            return chinese_text  # 查不到就回傳原本內容
    
    def _extract_translation_from_content(self, query: str, content: str) -> str:
        """從內容中提取翻譯"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if line.startswith('中文：') and query in line:
                if i + 1 < len(lines) and lines[i + 1].startswith('排灣語：'):
                    return lines[i + 1][4:].strip()
        
        return ""

# FastAPI 應用程式
app = FastAPI(title="排灣語雙向翻譯 API", description="排灣語與中文雙向翻譯服務")

# 全域翻譯器實例
paiwan_to_chinese_translator = None
chinese_to_paiwan_translator = None

@app.on_event("startup")
async def startup_event():
    global paiwan_to_chinese_translator, chinese_to_paiwan_translator
    paiwan_to_chinese_translator = PaiwanToChineseTranslator()
    chinese_to_paiwan_translator = ChineseToPaiwanTranslator()

@app.post("/translate/paiwan-to-chinese")
async def translate_paiwan_to_chinese(request: TranslateRequest, response: Response):
    """
    將排灣語翻譯成中文
    """
    # 設置標頭以禁用 keep-alive
    response.headers["Connection"] = "close"
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="輸入文字不能為空")
        
        result = paiwan_to_chinese_translator.translate(request.text)
        
        return {
            "tokens": request.text,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻譯過程發生錯誤: {str(e)}")

@app.post("/translate/chinese-to-paiwan")
async def translate_chinese_to_paiwan(request: TranslateRequest, response: Response):
    """
    將中文翻譯成排灣語
    """
    # 設置標頭以禁用 keep-alive
    response.headers["Connection"] = "close"
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="輸入文字不能為空")
        
        result = chinese_to_paiwan_translator.translate(request.text)
        
        return {
            "original": request.text,
            "translation": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻譯過程發生錯誤: {str(e)}")

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy", 
        "service": "paiwan_translation", 
        "endpoints": [
            "/translate/paiwan-to-chinese",
            "/translate/chinese-to-paiwan"
        ]
    }

@app.get("/")
async def root():
    """根端點"""
    return {
        "message": "排灣語雙向翻譯 API 服務",
        "endpoints": {
            "排灣語翻中文": "/translate/paiwan-to-chinese",
            "中文翻排灣語": "/translate/chinese-to-paiwan",
            "健康檢查": "/health",
            "API 文檔": "/docs"
        }
    }

if __name__ == "__main__":
    # 配置 Uvicorn 以禁用 keep-alive
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        # 設定 Uvicorn 伺服器不保持連接
        headers=[("Connection", "close")],
        # 設定較短的保持連接超時
        timeout_keep_alive=1
    )
