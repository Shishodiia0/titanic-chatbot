import os
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
app = FastAPI()

# -------------------------
# Load Dataset
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "titanic.csv")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower()

# -------------------------
# Initialize LLM + Agent
# -------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


prompt = PromptTemplate(
    input_variables=["question", "data_summary"],
    template="""
You are a professional data analyst working with the Titanic dataset.

Dataset Summary:
{data_summary}

User Question:
{question}

Rules:
- Answer only using the dataset.
- Do not generate Python code.
- Do not explain how to write code.
- Give a clear and concise analytical answer.
"""
)

chain = LLMChain(llm=llm, prompt=prompt)


# -------------------------
# Request / Response Models
# -------------------------
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    plot_image: Optional[str] = None

# -------------------------
# Helper: Generate Plot
# -------------------------
def generate_plot(question: str):
    question = question.lower()

    fig = None

    if "histogram" in question and "age" in question:
        fig = plt.figure()
        df["age"].dropna().hist()
        plt.title("Histogram of Passenger Ages")
        plt.xlabel("Age")
        plt.ylabel("Frequency")

    elif "embarked" in question:
        fig = plt.figure()
        df["embarked"].value_counts().plot(kind="bar")
        plt.title("Passengers by Embarkation Port")
        plt.xlabel("Port")
        plt.ylabel("Count")

    elif "male" in question or "female" in question:
        fig = plt.figure()
        df["sex"].value_counts().plot(kind="bar")
        plt.title("Passengers by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Count")

    if fig:
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return image_base64

    return None

# -------------------------
# Health Check
# -------------------------
@app.get("/")
def health_check():
    return {"status": "Backend is running"}

# -------------------------
# Chat Endpoint
# -------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        data_summary = df.describe(include="all").to_string()

        result = chain.invoke({
            "question": request.question,
            "data_summary": data_summary
        })

        answer = result["text"] if isinstance(result, dict) else result
        plot_image = generate_plot(request.question)

        return ChatResponse(
            answer=answer,
            plot_image=plot_image if plot_image else None
)

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "plot_image": None
        }