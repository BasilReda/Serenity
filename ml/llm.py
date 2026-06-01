from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint
from core.config import settings
import torch
import os
from dotenv import load_dotenv

load_dotenv()
def get_chat_model_hf():
    # We switch to a powerful 7B/8B model supported by HF Serverless
    repo_id = "Qwen/Qwen2.5-7B-Instruct" 
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=os.getenv("hf_api_key"),
        max_new_tokens=512,
        temperature=0.7, 
    )
    
    # ChatHuggingFace uses the repo_id internally to fetch the correct chat template
    return ChatHuggingFace(llm=llm, repo_id=repo_id)


def get_chat_dpo_local() -> ChatHuggingFace:
    tokenizer = AutoTokenizer.from_pretrained(settings.DPO_CHECKPOINT)
    model = AutoModelForCausalLM.from_pretrained(
        settings.DPO_CHECKPOINT,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=512, temperature=0.4, do_sample=True,
        pad_token_id=tokenizer.eos_token_id, return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

_global_llm = None
_dpo_model  = None

def get_global_llm() -> ChatHuggingFace:
    global _global_llm
    if _global_llm is None:
        print("[LLM] Loading global LLM (Qwen Hugging Face)...")
        _global_llm = get_chat_model_hf()
    return _global_llm

def get_dpo_model() -> ChatHuggingFace:
    global _dpo_model
    if _dpo_model is None:
        print("[LLM] Loading DPO local model...")
        _dpo_model = get_chat_dpo_local()
    return _dpo_model