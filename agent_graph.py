import os
import re
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph

from vector_store import VectorStoreManager


load_dotenv()


class AgentState(TypedDict, total=False):
    query: str
    route: str
    context: str
    web_context: str
    response: str
    sources: list[dict]
    confidence: str
    generated_from_llm: bool
    model_provider: str
    model_name: str
    temperature: float
    max_tokens: int
    top_p: float
    timings: dict[str, float]
    presentation_mode: bool


def _model_settings(state: AgentState):
    provider = (state.get("model_provider") or "gemini").lower()
    if provider not in {"gemini", "openai", "groq"}:
        provider = "gemini"

    if provider == "openai":
        default_model = "gpt-4o-mini"
    elif provider == "groq":
        default_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    else:
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model_name = state.get("model_name") or default_model
    temperature = float(state.get("temperature", 0.2))
    max_tokens = int(state.get("max_tokens", 1400))
    top_p = float(state.get("top_p", 0.9))

    # Groq free/on-demand tiers often enforce a strict tokens-per-minute budget.
    # Keep completion budget modest because prompt + evidence + output all count.
    max_token_limit = 1800 if provider == "groq" else 8000

    return {
        "provider": provider,
        "model_name": model_name,
        "temperature": max(0.0, min(2.0, temperature)),
        "max_tokens": max(256, min(max_token_limit, max_tokens)),
        "top_p": max(0.01, min(1.0, top_p)),
    }


def _build_llm(state: AgentState):
    settings = _model_settings(state)

    if settings["provider"] == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=settings["model_name"],
            api_key=api_key,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            max_completion_tokens=settings["max_tokens"],
        )

    if settings["provider"] == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        try:
            from langchain_groq import ChatGroq
        except Exception:
            return None
        return ChatGroq(
            model=settings["model_name"],
            api_key=api_key,
            temperature=settings["temperature"],
            max_tokens=settings["max_tokens"],
            model_kwargs={"top_p": settings["top_p"]},
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model=settings["model_name"],
        google_api_key=api_key,
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        max_tokens=settings["max_tokens"],
    )


def _build_web_search_tool():
    if not os.getenv("TAVILY_API_KEY"):
        return None
    return TavilySearch(max_results=5)


db_manager = VectorStoreManager()
web_search_tool = _build_web_search_tool()


def _is_generation_request(query):
    query = query.lower()
    generation_phrases = [
        "generate",
        "create",
        "make",
        "prepare",
        "build",
        "draft",
        "write",
        "portfolio",
        "ppt",
        "presentation",
        "deck",
        "report",
    ]
    return any(phrase in query for phrase in generation_phrases)


def _is_market_portfolio_request(query):
    normalized = query.lower()
    return "market portfolio" in normalized or ("portfolio" in normalized and any(
        word in normalized for word in ["market", "company", "business", "financial", "revenue"]
    ))


def _needs_current_financials(query):
    normalized = query.lower()
    finance_terms = [
        "financial",
        "revenue",
        "profit",
        "profitability",
        "growth",
        "margin",
        "market position",
        "market share",
        "valuation",
        "earnings",
        "annual report",
        "10-k",
        "quarter",
    ]
    return _is_market_portfolio_request(query) or any(term in normalized for term in finance_terms)


def _financial_search_query(query):
    clean = re.sub(r"\s+", " ", query).strip()
    if not _needs_current_financials(clean):
        return clean
    return (
        f"{clean} latest annual report investor relations revenue profitability growth "
        "gross margin net income free cash flow market position competitors"
    )


def _trim_context_for_model(context, provider, presentation_mode=False):
    clean = re.sub(r"\n{3,}", "\n\n", context or "").strip()
    if not clean:
        return ""

    if provider == "groq":
        limit = 4200 if presentation_mode else 5200
    else:
        limit = 12000 if presentation_mode else 9000

    if len(clean) <= limit:
        return clean

    blocks = [block.strip() for block in clean.split("\n\n") if block.strip()]
    selected = []
    current_len = 0
    for block in blocks:
        compact_block = re.sub(r"\s+", " ", block).strip()
        if current_len + len(compact_block) + 2 > limit:
            remaining = limit - current_len - 2
            if remaining > 500:
                selected.append(compact_block[:remaining].rsplit(" ", 1)[0])
            break
        selected.append(compact_block)
        current_len += len(compact_block) + 2
    return "\n\n".join(selected).strip()


def _format_source(metadata, score=None):
    page = metadata.get("page")
    page_label = f"page {page}" if page else "source"
    return {
        "source": metadata.get("source", "Unknown"),
        "page": page,
        "chunk": metadata.get("chunk"),
        "type": metadata.get("type"),
        "score": score,
        "label": f"{metadata.get('source', 'Unknown')} ({page_label})",
    }


def _timing_update(state: AgentState, step: str, started_at: float):
    timings = dict(state.get("timings", {}))
    timings[step] = round(time.perf_counter() - started_at, 4)
    return {"timings": timings}


def retrieval_node(state: AgentState):
    started_at = time.perf_counter()
    print("[Retrieval Agent] Searching local document memory...")
    results = db_manager.search(state["query"], k=5)

    if not results:
        return {"context": "", "sources": [], "confidence": "low", **_timing_update(state, "retrieval", started_at)}

    context_blocks = []
    sources = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        source = _format_source(metadata, result.get("score"))
        sources.append(source)
        context_blocks.append(
            f"[Source {index}: {source['label']}]\n{result.get('content', '')}"
        )

    best_score = results[0].get("score", 999)

    if best_score <= 0.4:
        confidence = "high"
    elif best_score <= 0.75:
        confidence = "medium"
    else:
        confidence = "low"  # Clear flag that local data is mathematically irrelevant

    return {
        "context": "\n\n".join(context_blocks),
        "sources": sources,
        "confidence": confidence,
        **_timing_update(state, "retrieval", started_at),
    }


def router_node(state: AgentState):
    """Uses the LLM to decide whether retrieved documents are sufficient."""
    started_at = time.perf_counter()
    query = state["query"]
    is_generation_request = _is_generation_request(query)
    current_llm = _build_llm(state)
    settings = _model_settings(state)

    if current_llm is None:
        has_context = bool(state.get("context", "").strip())
        return {
            "route": "generate" if has_context or is_generation_request else "web_search",
            **_timing_update(state, "router", started_at),
        }

    context = _trim_context_for_model(state.get("context", "").strip(), settings["provider"])

    if not context:
        print("[Router Agent] No local documents found.")
        if _needs_current_financials(query) and web_search_tool is not None:
            print("[Router Agent] Current financial request detected. Using web evidence.")
            return {"route": "web_search", **_timing_update(state, "router", started_at)}
        if is_generation_request:
            print("[Router Agent] Generation request detected. Using LLM general knowledge.")
            return {"route": "generate", **_timing_update(state, "router", started_at)}
        return {"route": "web_search", **_timing_update(state, "router", started_at)}

    prompt = f"""
You are a retrieval evaluator.

Question:
{state["query"]}

Retrieved Context:
{context}

Determine whether the retrieved context contains enough information
to answer the user's question.

Reply with ONLY ONE WORD.

SUFFICIENT

or

INSUFFICIENT
"""

    decision = current_llm.invoke(prompt).content.strip().upper()

    print(f"[Router Agent] Decision: {decision}")

    if "INSUFFICIENT" in decision:
        if _needs_current_financials(query) and web_search_tool is not None:
            print("[Router Agent] Retrieved context is weak for current financials. Using web evidence.")
            return {"route": "web_search", **_timing_update(state, "router", started_at)}
        if is_generation_request:
            print("[Router Agent] Retrieved context is weak. Using LLM general knowledge.")
            return {"route": "generate", **_timing_update(state, "router", started_at)}
        return {"route": "web_search", **_timing_update(state, "router", started_at)}

    return {"route": "generate", **_timing_update(state, "router", started_at)}
def web_search_node(state: AgentState):
    started_at = time.perf_counter()
    if web_search_tool is None:
        return {
            "web_context": "",
            "sources": state.get("sources", []),
            "confidence": "low",
            **_timing_update(state, "web_search", started_at),
        }

    search_query = _financial_search_query(state["query"])
    print("[Web Search Agent] Searching external sources...")
    search_results = web_search_tool.invoke({"query": search_query})

    web_blocks = []
    web_sources = []
    if isinstance(search_results, list):
        for index, result in enumerate(search_results, start=1):
            if not isinstance(result, dict):
                continue
            title = result.get("title") or result.get("url") or f"Web result {index}"
            content = result.get("content", "")
            url = result.get("url")
            web_blocks.append(f"[Web {index}: {title}]\n{content}")
            web_sources.append({
                "source": title,
                "url": url,
                "page": None,
                "chunk": None,
                "score": None,
                "label": title,
            })
    else:
        web_blocks.append(str(search_results))

    return {
        "web_context": "\n\n".join(web_blocks),
        "sources": [*state.get("sources", []), *web_sources],
        "confidence": "medium" if web_blocks else "low",
        **_timing_update(state, "web_search", started_at),
    }


def _fallback_answer(state: AgentState):
    context = state.get("context") or state.get("web_context") or ""

    if not context.strip():
        return (
            "I could not find relevant information in the indexed documents, "
            "and the LLM service is not configured to generate a fallback answer."
        )

    excerpts = []

    for block in context.split("\n\n")[:3]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            excerpts.append(" ".join(lines[:3]))

    return (
        "Here is the most relevant information I found:\n\n"
        + "\n\n".join(excerpts)
    )


def generator_node(state: AgentState):
    started_at = time.perf_counter()
    print("[Response Generator] Creating final answer...")
    current_llm = _build_llm(state)
    settings = _model_settings(state)
    combined_context = "\n\n".join(
        value for value in [state.get("context", ""), state.get("web_context", "")] if value.strip()
    )
    has_evidence = bool(combined_context.strip()) and state.get("confidence") != "low"
    presentation_mode = bool(state.get("presentation_mode"))
    evidence_context = _trim_context_for_model(combined_context, settings["provider"], presentation_mode)

    if current_llm is None:
        if settings["provider"] == "openai":
            provider_name = "OpenAI"
            missing_key = "OPENAI_API_KEY"
        elif settings["provider"] == "groq":
            provider_name = "Groq"
            missing_key = "GROQ_API_KEY or the langchain-groq package"
        else:
            provider_name = "Gemini"
            missing_key = "GOOGLE_API_KEY"
        return {
            "response": f"{provider_name} is selected, but {missing_key} is not configured. Add the key to .env or switch model provider.",
            "generated_from_llm": True,
            "confidence": "missing_key",
            "sources": [],
            **_timing_update(state, "generator", started_at),
        }

    if presentation_mode and settings["provider"] == "groq":
        prompt = f"""
You are a senior business analyst creating a slide-ready market portfolio draft.

Use evidence when relevant. If evidence is weak, use general business knowledge, but do not invent exact financial numbers.
For public companies, financial values must include a period/source label or say "Verify from latest filings".
Write concise, non-repetitive content for an 8-slide PPT.

Required output order and exact headings:

Market Portfolio:
- Title line: Market Portfolio for <Company or Platform>

Company Overview:
- Markdown table: Company | Industry | Target Customers | Core Value Proposition

Financial Snapshot:
- Markdown table exactly: Metric | Latest Value | Period / Source | Business Meaning
- 5 rows: revenue, revenue growth, profitability/net income, gross margin, free cash flow/cash/customer signal

Product Portfolio:
- Markdown table: Category | Products | Business Role | Priority
- 5 to 6 rows only

Market Segments And Geographic Reach:
- Markdown table: Segment | Description | Primary Need | Commercial Potential
- 4 to 5 rows only

Competitive Landscape And Positioning:
- Markdown table: Competitor | Strength | Weakness Compared To The Company | Relative Threat
- 4 competitors only

Growth Opportunities And Strategic Priorities:
- 4 concise bullets with action, KPI/signal, and business impact

References:
- List source names only. If none were used, write: LLM general knowledge

Rules:
- Keep the full response under 1200 words.
- Prefer numbers, periods, margins, revenue/growth/profitability signals when evidenced.
- Label uncertain values as estimates or "Verify from latest filings".
- Do not mention retrieval failure or irrelevant evidence.
- Do not use citations in the main body.

User question:
{state["query"]}

Collected evidence:
{evidence_context if has_evidence else "No relevant evidence was found."}

Final draft:
"""
    elif presentation_mode:
        prompt = f"""
You are a senior business analyst preparing a presentation-ready market portfolio draft.
The draft will be reviewed by the user first and then converted into a PowerPoint deck.

Use collected evidence only when it directly supports the request. If the evidence is missing,
weak, stale, or unrelated, create the analysis from your general business knowledge instead.
Do not mention irrelevant evidence, retrieval failure, or phrases like "the provided evidence discusses",
"the provided evidence does not", "I could not find", or "based on limited information".
If retrieved context is from a generated_knowledge source, treat it as reusable notes and rewrite it cleanly.

Quality bar:
- Write for a mentor/reviewer expecting business-quality analysis, not a short chatbot answer.
- Make the output slide-ready: crisp headings, dense bullets, specific business language, no filler.
- Prefer analysis over description: explain market logic, competitive implications, risks, and recommendations.
- Include realistic qualitative assessments when exact numbers are unavailable; do not invent precise fake statistics.
- For public companies, prioritize latest reported company/investor evidence for revenue, profitability, margins, growth,
  cash flow, customer count, ARPU, and guidance before using general knowledge.
- Separate exact reported figures from estimates. Every financial number must include a period such as FY2025, Q1 FY2026,
  TTM, or "estimate" when the period is not verifiable from evidence.
- If web/company evidence is unavailable, do not pretend the numbers are current; use "verify from latest filings" in the
  value or period field and still explain what the metric means.
- Make the analysis technical, statistical, and numbers-oriented wherever possible.
- Include measurable indicators such as customer segments, service categories, pricing/value signals, growth rates, adoption metrics,
  revenue contribution levels, risk severity, and KPI targets when they are supported by evidence or clearly framed as estimates.
- Use approximate ranges such as "low/medium/high", "3-5 year horizon", "4/5", "5-7 priority services", or "20-30% relative improvement"
  only when exact data is unavailable; label uncertain values as estimates.
- Do not fabricate exact revenue, customer count, market share, or financial figures. If a precise number is not known, use directional
  ranges, ratings, or "verify from latest filings/company sources".
- For a company/platform request, cover customers, product portfolio, competitors, pricing/value, growth opportunities,
  risks, and actionable next steps.
- Avoid repeating the same idea across sections.
- Every bullet must contain a specific claim plus a business implication.
- The answer must be complete enough to generate a PPT directly without the user asking for refinement.
- Use concise but information-dense wording; each bullet should be 18 to 32 words.
- Include table-ready business analysis, not generic labels.

Output contract:
- Use the exact headings below and keep the exact order.
- Do not add extra headings before Executive Summary.
- Do not wrap the answer in markdown code blocks.
- Do not use citations inside the main body; keep references only under References.
- Before returning, silently verify that Suggested Table has exactly six usable data rows.

If the user asks for a "market portfolio" for a company or platform, create a concise mentor-presentation draft.
The goal is not a long report. The goal is a complete, presentable deck with about 8 to 10 slides including cover and references.

Market Portfolio:
- Start with a clear title line: Market Portfolio for <Company or Platform>

Company Overview:
- Include a compact markdown table with: Company, Industry, Target Customers, Core Value Proposition
- If exact founded/headquarters details are uncertain, use "Verify from latest company sources" rather than inventing precision
- Include measurable business context where possible, such as public company status, primary cloud category, or customer-size focus

Financial Snapshot:
- Include a markdown table with exactly this header:
  | Metric | Latest Value | Period / Source | Business Meaning |
  | --- | --- | --- | --- |
- Include 5 to 7 rows covering revenue, revenue growth, profitability or net income, gross margin, free cash flow or cash position,
  customer/ARPU signal, and guidance or market valuation when available
- Use exact reported numbers only when available from collected evidence; otherwise write "Verify from latest filings" for the value
- Keep each Business Meaning cell analytical and concise, explaining what the number says about scale, efficiency, growth, or risk

Product Portfolio:
- Include a markdown table with columns: Category | Products | Business Role | Priority
- Cover only the 5 to 7 most important categories for the presentation
- Priority should be High, Medium, or Emerging

Market Segments And Geographic Reach:
- Include a markdown table with columns: Segment | Description | Primary Need | Commercial Potential
- Include 4 to 6 important segments
- Add 2 bullets after the table about geographic reach, latency, or international expansion implications

Competitive Landscape And Positioning:
- Include a markdown table with columns: Competitor | Strength | Weakness Compared To The Company | Relative Threat
- Include 4 to 5 likely competitors
- Add 2 bullets explaining the company's positioning versus hyperscalers and lower-cost alternatives

SWOT Analysis:
- Use four subheadings: Strengths, Weaknesses, Opportunities, Threats
- Put 2 to 3 bullets under each subheading
- Add a severity/impact word in each bullet where useful, such as High, Medium, or Low

Customer Value Proposition:
- Use 5 to 6 bullets focused on customer value, not feature listing
- Include measurable value language such as reduced setup time, predictable billing, faster deployment, lower operational overhead, or retention impact

Revenue And Go-To-Market:
- Include a markdown table with columns: Area | Strategy | Metric / Signal | Why It Matters
- Combine revenue streams and 4Ps into 5 to 6 rows

Growth Opportunities And Strategic Priorities:
- Use 5 to 6 bullets with specific growth paths, recommendations, expected impact, and measurable KPI

Conclusion:
- Write one strong paragraph summarizing the market niche, strategic advantage, risks, and future opportunity

References:
- If no document or web evidence was used, write only: LLM general knowledge
- Otherwise list source names only

For the market-portfolio presentation structure, do not use the shorter section list below and do not create more than these sections.

Executive Summary:
- 4 strong bullets
- each bullet should summarize a major market insight, implication, or recommendation
- include the overall opportunity, positioning, and key risk/recommendation

Market Position:
- 5 bullets on target customers, category position, differentiation, pricing/value thesis, and competitive positioning
- mention likely competitors or alternatives when relevant

Portfolio Overview:
- 6 to 8 numbered items describing offerings, capabilities, or portfolio components
- for each item, include what it is, target user/use case, and why it matters commercially

Comparative Analysis:
- 5 to 6 bullets comparing strengths, gaps, risks, and competitive factors
- compare against likely alternatives, market expectations, or customer buying criteria
- include at least one risk/tradeoff and one defensible advantage

Strategic Recommendations:
- 5 practical recommendations with actions, rationale, and expected impact
- include at least one go-to-market recommendation and one product/portfolio recommendation

Suggested Table:
- Create a markdown table with exactly this header:
  | Segment | Offering | Customer Value | Strategic Importance |
  | --- | --- | --- | --- |
- Include exactly 6 rows
- Each row must have all 4 columns filled
- Each row must be specific, non-repetitive, and useful as a slide table
- Customer Value and Strategic Importance must be analytical phrases of 8 to 16 words, not one-word labels
- Do not put bullet points inside table cells
- Example row style:
  | Startups | Core cloud compute | Predictable launch infrastructure with lower setup complexity | Builds entry-level adoption and future expansion paths |

Risks And Mitigations:
- 3 to 4 bullets listing key market/product/execution risks and how to reduce them

Success Metrics:
- 3 to 5 measurable indicators the business should track, such as adoption, conversion, retention, revenue mix,
  usage depth, or customer expansion

References:
- If no document or web evidence was used, write only: LLM general knowledge
- Otherwise list source names only

User question:
{state["query"]}

Collected evidence:
{evidence_context if has_evidence else "No relevant evidence was found."}

Final draft:
"""
    else:
        prompt = f"""
You are an Agentic Document Intelligence assistant and senior analyst.

Answer the user's question using collected evidence only when it directly supports the request.
If no relevant evidence is available, generate a complete answer from your general knowledge and clearly mark
the source as "LLM general knowledge" in the Sources section.
Do not discuss irrelevant evidence, retrieval failure, or phrases like "the provided evidence discusses",
"the provided evidence does not", "I could not find", or "based on limited information".
If retrieved context is from a generated_knowledge source, treat it as reusable notes and rewrite it cleanly.
Prefer local document evidence over web evidence when both are useful.

Keep the answer clear, structured, and mentor-ready:
- start with a direct answer or executive summary
- include concrete analysis, tradeoffs, risks, and recommendations when the question is business-oriented
- use bullets and tables when they improve clarity
- avoid repetitive phrasing and generic claims
- do not invent exact numbers; use qualitative estimates when necessary

End with a short "Sources" section. If no document or web evidence was used, write "LLM general knowledge".

User question:
{state["query"]}

Collected evidence:
{evidence_context if has_evidence else "No relevant evidence was found."}

Final answer:
"""
    response = current_llm.invoke(prompt)
    updates = {
        "response": response.content,
        "generated_from_llm": not has_evidence,
        **_timing_update(state, "generator", started_at),
    }

    if not has_evidence:
        updates["sources"] = [{
            "source": "LLM general knowledge",
            "page": None,
            "chunk": None,
            "score": None,
            "label": "LLM general knowledge",
        }]
        updates["confidence"] = "llm"

    return updates


workflow = StateGraph(AgentState)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("router", router_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("retrieval")
workflow.add_edge("retrieval", "router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "generate": "generator",
        "web_search": "web_search",
    },
)
workflow.add_edge("web_search", "generator")
workflow.add_edge("generator", END)

app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"query": "What are the key policies for interns?"})
    print(result["response"])
