import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser

from urllib.parse import quote_plus
from typing import TypedDict

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, START, END

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it to your .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_retries=2
)

def get_stock_data(ticker, period="6mo"):
    df = yf.Ticker(ticker).history(
        period=period,
        auto_adjust=True
    )

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    df = df.dropna().copy()

    df["Daily_Return_%"] = (
        df["Close"].pct_change() * 100
    )

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["Volatility_20"] = (
        df["Daily_Return_%"].rolling(20).std()
        * np.sqrt(252)
    )

    return df

def stock_data_agent(state):
    try:
        df = get_stock_data(state["ticker"])

        latest = df.iloc[-1]
        close = float(latest["Close"])
        initial_close = float(df["Close"].iloc[0])

        period_return = (
            close / initial_close - 1
        ) * 100

        sma20 = latest["SMA_20"]
        sma50 = latest["SMA_50"]
        volatility = latest["Volatility_20"]

        market_data = {
            "ticker": state["ticker"],
            "latest_date": str(df.index[-1].date()),
            "latest_close": round(close, 2),
            "period_return_pct": round(period_return, 2),
            "sma20": (
                round(float(sma20), 2)
                if pd.notna(sma20) else None
            ),
            "sma50": (
                round(float(sma50), 2)
                if pd.notna(sma50) else None
            ),
            "annualized_volatility_pct": (
                round(float(volatility) * 100, 2)
                if pd.notna(volatility) else None
            ),
            "observations": len(df)
        }

        return {
            "market_data": market_data,
            "data_ok": len(df) >= 50,
            "data_error": ""
        }

    except Exception as exc:
        return {
            "market_data": {},
            "data_ok": False,
            "data_error": str(exc)
        }

    def get_financial_news(company_name, max_items=10):
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(
            f'"{company_name}" stock financial results'
        )
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries[:max_items]:
        articles.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", "")
        })

    return articles

def build_news_index(articles):
    documents = []

    for article in articles:
        content = (
            f"Headline: {article['title']}\n"
            f"Published: {article['published']}\n"
            f"Summary: {article['summary']}\n"
            f"Source: {article['link']}"
        )

        if article["title"] or article["summary"]:
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": article["link"],
                        "title": article["title"],
                        "published": article["published"]
                    }
                )
            )

    if not documents:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(documents, embeddings)


def retrieve_relevant_news(query, index, k=4):
    if index is None:
        return []

    return index.similarity_search(query, k=k)

def news_researcher_agent(state):
    try:
        articles = get_financial_news(
            state["company_name"]
        )

        index = build_news_index(articles)

        docs = retrieve_relevant_news(
            (
                f"{state['company_name']} financial results, "
                "earnings, business performance and risks"
            ),
            index,
            k=4
        )

        evidence = "\n\n".join(
            doc.page_content for doc in docs
        )

        return {
            "news_articles": articles,
            "news_evidence": evidence,
            "news_ok": bool(docs),
            "news_error": ""
        }

    except Exception as exc:
        return {
            "news_articles": [],
            "news_evidence": "",
            "news_ok": False,
            "news_error": str(exc)
        }

    class StockAssistantState(TypedDict, total=False):
    ticker: str
    company_name: str

    market_data: dict
    data_ok: bool
    data_error: str

    news_articles: list
    news_evidence: str
    news_ok: bool
    news_error: str

    insight_report: str
    recommendation: str
    risk_report: str
    final_report: str

    insight_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a cautious financial research analyst.

        Use only the supplied data and news evidence.
        Separate facts from interpretations.
        Never invent prices, news, or financial figures.
        Mention missing information and uncertainty.
        This is an educational simulation, not financial advice."""
    ),
    (
        "human",
        """Stock: {ticker}

        Historical market data:
        {market_data}

        Retrieved news evidence:
        {news_evidence}

        Analyze:
        1. Historical performance
        2. Moving-average trends
        3. News-related developments
        4. Potential opportunities
        5. Risks and limitations"""
    )
])

insight_chain = (
    insight_prompt | llm | StrOutputParser()
)


def insight_analyst_agent(state):
    report = insight_chain.invoke({
        "ticker": state["ticker"],
        "market_data": json.dumps(
            state.get("market_data", {}),
            indent=2
        ),
        "news_evidence": (
            state.get("news_evidence")
            or "No relevant news evidence was retrieved."
        )
    })

    return {"insight_report": report}

recommendation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a cautious financial risk analyst.

        Return an educational BUY, SELL, or HOLD assessment.
        This is not financial advice.
        Never guarantee future prices or returns.
        Do not invent evidence or numerical probabilities.
        Choose HOLD if evidence is insufficient or contradictory."""
    ),
    (
        "human",
        """Stock: {ticker}

        Historical market data:
        {market_data}

        Research report:
        {insight_report}

        Return:
        Recommendation: BUY, SELL, or HOLD
        Justification:
        Supporting evidence:
        Key risks:
        Missing information:
        Confidence: HIGH, MEDIUM, or LOW, with explanation"""
    )
])

recommendation_chain = (
    recommendation_prompt | llm | StrOutputParser()
)


def risk_recommendation_agent(state):
    if not state.get("data_ok", False):
        return {
            "recommendation": "HOLD",
            "risk_report": (
                "HOLD — insufficient historical data. "
                "No directional recommendation is justified."
            )
        }

    result = recommendation_chain.invoke({
        "ticker": state["ticker"],
        "market_data": json.dumps(
            state.get("market_data", {}),
            indent=2
        ),
        "insight_report": state.get("insight_report", "")
    })

    return {
        "recommendation": result,
        "risk_report": result
    }

def insufficient_data_agent(state):
    return {
        "recommendation": "HOLD",
        "risk_report": (
            "HOLD — historical data is insufficient. "
            f"Details: {state.get('data_error', 'Unknown error')}"
        ),
        "insight_report": (
            "Analysis stopped because adequate historical "
            "data was unavailable."
        )
    }


def compile_final_report(state):
    report = f"""
AI STOCK MARKET ASSISTANT
=========================

Stock: {state.get('ticker', 'Unknown')}

MARKET DATA
-----------
{json.dumps(state.get('market_data', {}), indent=2)}

AI INSIGHTS
-----------
{state.get('insight_report', 'Not available')}

SIMULATED RECOMMENDATION
------------------------
{state.get('recommendation', 'HOLD')}

RISK ANALYSIS
-------------
{state.get('risk_report', 'Not available')}

NEWS EVIDENCE
-------------
{state.get('news_evidence', 'No relevant news retrieved')}

NEWS RETRIEVAL STATUS
---------------------
{state.get('news_error', 'No reported retrieval error')}

DISCLAIMER
----------
Educational simulation only. Not financial advice.
"""

    return {"final_report": report}

def route_after_data(state):
    if state.get("data_ok", False):
        return "analyze"

    return "insufficient"


graph_builder = StateGraph(StockAssistantState)

graph_builder.add_node("stock_data", stock_data_agent)
graph_builder.add_node("news_research", news_researcher_agent)
graph_builder.add_node("insights", insight_analyst_agent)
graph_builder.add_node("recommendation", risk_recommendation_agent)
graph_builder.add_node("insufficient", insufficient_data_agent)
graph_builder.add_node("final_report", compile_final_report)

graph_builder.add_edge(START, "stock_data")
graph_builder.add_edge("stock_data", "news_research")

graph_builder.add_conditional_edges(
    "news_research",
    route_after_data,
    {
        "analyze": "insights",
        "insufficient": "insufficient"
    }
)

graph_builder.add_edge("insights", "recommendation")
graph_builder.add_edge("recommendation", "final_report")
graph_builder.add_edge("insufficient", "final_report")
graph_builder.add_edge("final_report", END)

stock_graph = graph_builder.compile()

print("LangGraph compiled successfully!")

from dotenv import load_dotenv
load_dotenv()

from stock_assistant import stock_graph

result = stock_graph.invoke({
    "ticker": "RELIANCE.NS",
    "company_name": "Reliance Industries"
})

print(result["final_report"])