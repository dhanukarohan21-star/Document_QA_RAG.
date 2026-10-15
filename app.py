
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from stock_assistant import stock_graph

st.set_page_config(
	page_title="AI Stock Market Assistant",
	page_icon="📈",
	layout="wide"
)

st.title("AI Stock Market Assistant")
st.caption("LangChain + LangGraph + Gemini + RAG")

st.warning(
	"Educational simulation only. "
	"This is not financial advice."
)

with st.sidebar:
	st.header("Stock Configuration")

	ticker = st.text_input(
		"Stock ticker",
		value="RELIANCE.NS"
	).strip().upper()

	company = st.text_input(
		"Company name",
		value="Reliance Industries"
	).strip()

	analyze = st.button(
		"Analyze Stock",
		type="primary"
	)

if analyze:
	if not ticker or not company:
		st.error("Enter both a ticker and company name.")

	else:
		try:
			with st.spinner(
				"Running stock data, news, and AI agents..."
			):
				result = stock_graph.invoke({
					"ticker": ticker,
					"company_name": company
				})

			market = result.get("market_data", {})

			if market:
				col1, col2, col3 = st.columns(3)

				col1.metric(
					"Latest historical close",
					f"{market['latest_close']:.2f}"
				)

				col2.metric(
					"Historical period return",
					f"{market['period_return_pct']:.2f}%"
				)

				col3.metric(
					"Observations",
					market.get("observations", 0)
				)

				st.caption(
					"Latest available date: "
					+ market.get("latest_date", "Unknown")
				)

			st.subheader("AI Insights")
			st.write(result.get("insight_report", ""))

			st.subheader("Simulated Recommendation")
			st.write(result.get("recommendation", "HOLD"))

			st.subheader("Risk Analysis")
			st.write(result.get("risk_report", ""))

			st.subheader("Final Analysis Report")
			st.text(result.get("final_report", ""))

			st.subheader("Retrieved Financial News")

			articles = result.get("news_articles", [])

			if articles:
				for article in articles:
					st.markdown(
						f"- [{article['title']}]"
						f"({article['link']})"
					)
			else:
				st.info("No news articles were retrieved.")

		except Exception as exc:
			st.error(
				"Analysis failed. Check the ticker, "
				"internet connection, API key, and logs."
			)
			st.exception(exc)
