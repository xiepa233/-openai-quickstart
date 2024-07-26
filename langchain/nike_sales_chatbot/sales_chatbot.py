import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS


def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(base_url="https://api.xiaoai.plus/v1"),allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,base_url="https://api.xiaoai.plus/v1")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,base_url="https://api.xiaoai.plus/v1")
    ans = SALES_BOT({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        prompt = f"作为一个Nike销售专家，如果客户问我：'{message}'，我应该怎样回答呢？"
        response = llm.generate(prompt)
        print(f"[generated_response]{response}")
        return response
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="Nike销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
