import json
from typing import Dict
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
        # 初始化翻譯器，載入詞彙字典
        self.collection_name = collection_name
        
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
            if similarity > 80 and word != paiwan_text and similarity >= best_score:
                if similarity > best_score:
                    best_score = similarity
                    all_translations = list(translations)
                elif similarity == best_score:
                    all_translations.extend(translations)
        
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


# FastAPI 應用程式
app = FastAPI(title="排灣語雙向翻譯 API", description="排灣語與中文雙向翻譯服務")

# 全域翻譯器實例
paiwan_to_chinese_translator = None
chinese_to_paiwan_translator = None

@app.on_event("startup")
async def startup_event():
    global paiwan_to_chinese_translator
    paiwan_to_chinese_translator = PaiwanToChineseTranslator()

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


@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy", 
        "service": "paiwan_translation", 
        "endpoints": [
            "/translate/paiwan-to-chinese"
        ]
    }

@app.get("/")
async def root():
    """根端點"""
    return {
        "message": "排灣語雙向翻譯 API 服務",
        "endpoints": {
            "排灣語翻中文": "/translate/paiwan-to-chinese",
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
