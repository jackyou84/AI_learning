import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")

def initialize_sales_bot(vector_store_dir: str="estates_sale"):
    sale_type_list = ["房产销售", "教育咨询", "电器销售", "家装销售"]
    num = 0
    for data_file in ["estate_sales_data.txt","education_data.txt","electric_appliance_sales_data.txt","decoration_data.txt"]:
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found.")
        # 从文件加载数据
        docs = TextLoader(data_file).load()

        # FAISS 向量数据库，使用 docs 的向量作为初始化存储
        db = FAISS.from_documents(docs, OpenAIEmbeddings(base_url=base_url, api_key=api_key))
        #db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0
            )

        global SALES_BOT
        SALES_BOT[sale_type_list[num]] = RetrievalQA.from_chain_type(llm,
                                                retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                        search_kwargs={"score_threshold": 0.8}))
        # 返回向量数据库的检索结果
        SALES_BOT[sale_type_list[num]].return_source_documents = True
        num += 1

    return SALES_BOT

def sales_chat(sale_type, message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    print(f"[sale_type]{sale_type}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT[sale_type]({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "抱歉！这个问题暂时无法解答！"
    

def launch_gradio():
    demo = gr.Interface(
        fn=sales_chat,
        title="销售机器人",
        inputs=[
            gr.dropdown(["房产销售", "电器销售", "家装销售", "教育咨询"], label="销售场景"),
        ],
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
