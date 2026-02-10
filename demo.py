"""
Deep Research Agent — Streamlit Demo UI.

Full-featured research tool:
- Accept query, paper upload, or URL
- Deep research using arXiv + web search
- Downloadable PDF report
- Follow-up chat with context

Run: streamlit run demo.py
"""

import streamlit as st
import time
import os
import sys
import tempfile

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.deep_research import DeepResearchPipeline, BaselineAgent


# ─── Page Config ──────────────────────────────────────────

st.set_page_config(
    page_title="🔬 Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────

st.markdown("""
<style>
    .paradigm-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-a1 { background: #FFE0B2; color: #E65100; }
    .badge-a2 { background: #E3F2FD; color: #1565C0; }
    .badge-t1 { background: #E8F5E9; color: #2E7D32; }
    .badge-t2 { background: #F3E5F5; color: #7B1FA2; }
    .badge-none { background: #EEEEEE; color: #616161; }
    .step-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        background: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ──────────────────────────────────────────

if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ─── Sidebar ──────────────────────────────────────────

with st.sidebar:
    st.header("🧬 Paradigm Reference")
    
    with st.expander("View Paradigms", expanded=False):
        st.markdown("""
        <div class="step-card">
            <span class="paradigm-badge badge-a1">A1</span>
            <strong>Agent ← Tool Signal</strong><br>
            <small>Agent learns from tool feedback</small>
        </div>
        <div class="step-card">
            <span class="paradigm-badge badge-a2">A2</span>
            <strong>Agent ← Output Signal</strong><br>
            <small>Agent learns from output quality</small>
        </div>
        <div class="step-card">
            <span class="paradigm-badge badge-t1">T1</span>
            <strong>Tool (Independent)</strong><br>
            <small>Pre-trained, plug-and-play tool</small>
        </div>
        <div class="step-card">
            <span class="paradigm-badge badge-t2">T2</span>
            <strong>Tool ← Agent Signal</strong><br>
            <small>Tool optimized for frozen agent</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    st.header("⚙️ Settings")
    provider = st.selectbox("LLM Provider", ["groq", "gemini"], index=0)
    
    if provider == "groq":
        api_key = st.text_input("Groq API Key", type="password",
                                value=os.environ.get("GROQ_API_KEY", ""))
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
    else:
        api_key = st.text_input("Gemini API Key", type="password",
                                value=os.environ.get("GEMINI_API_KEY", ""))
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

    # Memory
    st.markdown("---")
    st.header("🧠 Memory (T2)")
    from core.memory import PersistentMemory
    memory = PersistentMemory()
    past_queries = memory.get_all_queries()
    if past_queries:
        st.success(f"{len(past_queries)} past sessions")
        for q in past_queries[-5:]:
            st.caption(f"• {q[:50]}")
    else:
        st.info("No past research yet")


# ─── Main Content ──────────────────────────────────────────

st.title("🔬 Deep Research Agent")

# ─── Input Section ──────────────────────────────────────────

st.subheader("📥 Input")

input_tab1, input_tab2, input_tab3 = st.tabs([
    "📝 Query Only", "📄 Upload Paper", "🔗 Paste URL"
])

with input_tab1:
    query = st.text_input(
        "Research Query",
        value="",
        placeholder="e.g., Recent advances in LLM tool use and function calling",
        key="query_only"
    )
    paper_path = None
    url_input = None

with input_tab2:
    query_with_paper = st.text_input(
        "Research Query (about the paper)",
        value="",
        placeholder="e.g., Explain the key contributions and compare with existing work",
        key="query_paper"
    )
    uploaded_file = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
        help="Upload a PDF paper — the system will analyze it + search for related work"
    )

with input_tab3:
    query_with_url = st.text_input(
        "Research Query (about the link)",
        value="",
        placeholder="e.g., Analyze this paper and find related research",
        key="query_url"
    )
    url_input_val = st.text_input(
        "URL",
        value="",
        placeholder="https://arxiv.org/abs/2503.00001",
        help="Paste any URL — web page, arXiv paper, blog post, etc."
    )


# ─── Run Research ──────────────────────────────────────────

if st.button("🚀 Start Deep Research", type="primary", use_container_width=True):
    
    if not api_key:
        st.error("Please enter your API key in the sidebar!")
        st.stop()

    # Determine which input mode
    final_query = query or query_with_paper or query_with_url
    final_paper_path = None
    final_url = None

    if not final_query:
        st.error("Please enter a research query!")
        st.stop()

    # Handle paper upload
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            final_paper_path = f.name

    # Handle URL
    if url_input_val:
        final_url = url_input_val

    # Reset state
    st.session_state.chat_history = []
    st.session_state.research_result = None

    # ─── Pipeline Execution ──────────────────────
    st.markdown("---")
    st.header("🔬 Research Pipeline")

    # Status containers
    step_containers = {
        "Input Processing": st.status("📥 Processing Input (T1)", expanded=False),
        "Memory Recall": st.status("🧠 Memory Recall (T2)", expanded=False),
        "Research Planning": st.status("📋 Step 1: Research Planning (A2)", expanded=True),
        "Iterative Search": st.status("🔍 Step 2: Search arXiv + Web (T1+T2)", expanded=False),
        "Self-Critique": st.status("🔎 Step 3: Self-Critique (A1+A2)", expanded=False),
        "Synthesis + Memory": st.status("✍️ Step 4: Synthesis + Memory (A2+T2)", expanded=False),
    }
    progress_bar = st.progress(0)

    def ui_callback(step_name, paradigm, status, details=None):
        step_order = ["Input Processing", "Memory Recall", "Research Planning",
                      "Iterative Search", "Self-Critique", "Synthesis + Memory"]
        if step_name in step_order:
            idx = step_order.index(step_name)
            progress_bar.progress((idx + 1) / len(step_order))

        container = step_containers.get(step_name)
        if container:
            with container:
                badges = " ".join(
                    f'<span class="paradigm-badge badge-{p.lower()}">{p}</span>'
                    for p in paradigm.split("+")
                )
                st.markdown(f"**Paradigm:** {badges}", unsafe_allow_html=True)
                st.write(f"Status: {status}")
                if details:
                    if "sub_questions" in details:
                        for sq in details["sub_questions"]:
                            st.write(f"  → {sq}")
                    if "total_papers" in details:
                        st.metric("Papers Found", details["total_papers"])
                    if "total_web_results" in details:
                        st.metric("Web Results", details["total_web_results"])
                    if "quality_score" in details:
                        st.metric("Quality Score", f"{details['quality_score']:.0%}")

    # Run pipeline
    try:
        pipeline = DeepResearchPipeline(provider=provider)
        st.session_state.pipeline = pipeline

        result = pipeline.research(
            query=final_query,
            paper_path=final_paper_path,
            url=final_url,
            callback=ui_callback
        )
        progress_bar.progress(1.0)
        st.session_state.research_result = result
        st.session_state.research_query = final_query

        # Cleanup temp file
        if final_paper_path and os.path.exists(final_paper_path):
            os.unlink(final_paper_path)

    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        st.exception(e)


# ─── Results Section ──────────────────────────────────────────

if st.session_state.research_result:
    result = st.session_state.research_result

    st.markdown("---")
    st.header("📄 Research Report")

    # Quality metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        quality = result["quality"]["overall"]
        st.metric("Quality Score", f"{quality:.0%}",
                  delta="Pass ✅" if quality >= 0.6 else "Low ⚠️")
    with col2:
        total_papers = sum(
            len(r.papers) for r in result["search_results"].values()
        )
        st.metric("Papers Found", total_papers)
    with col3:
        total_web = sum(
            len(r.web_results) for r in result["search_results"].values()
        )
        st.metric("Web Results", total_web)
    with col4:
        st.metric("Steps", len(result["pipeline_log"].steps))

    # Report
    st.markdown(result["report"])

    # ─── Download buttons ──────────────────────
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        # Download as Markdown
        st.download_button(
            "📥 Download Report (Markdown)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_dl2:
        # Download as PDF
        try:
            pipeline = st.session_state.pipeline
            if pipeline:
                pdf_path = pipeline.generate_pdf(
                    st.session_state.get("research_query", "Research"),
                    result
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Download Report (PDF)",
                        data=f.read(),
                        file_name="research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
        except Exception as e:
            st.caption(f"PDF generation unavailable: {e}")

    # ─── Pipeline Log ──────────────────────────
    with st.expander("🧬 Pipeline Log — Paradigms Used"):
        for step in result["pipeline_log"].steps:
            badges = " ".join(
                f'<span class="paradigm-badge badge-{p.lower()}">{p}</span>'
                for p in step.paradigm.split("+")
            )
            st.markdown(
                f"**{step.name}** {badges} — "
                f"*{step.duration:.1f}s* — {step.status}",
                unsafe_allow_html=True
            )

    # ─── Papers Found ──────────────────────────
    with st.expander(f"📚 Papers Found ({total_papers} papers, {total_web} web results)"):
        for sub_q, results in result["search_results"].items():
            st.markdown(f"**📌 {sub_q}** ({len(results.papers)} papers, "
                        f"{len(results.web_results)} web)")
            for paper in results.papers[:3]:
                st.markdown(
                    f"- **{paper.title}** [{paper.arxiv_id}]\n"
                    f"  {paper.abstract[:150]}..."
                )
            for wr in results.web_results[:2]:
                st.markdown(f"- 🌐 [{wr.title}]({wr.url})")
            st.markdown("---")

    # ─── Follow-up Chat ──────────────────────────
    st.markdown("---")
    st.header("💬 Follow-up Questions")
    st.caption("Ask questions about the research — the agent remembers everything")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if followup := st.chat_input("Ask a follow-up question about the research..."):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.markdown(followup)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pipeline = st.session_state.pipeline
                if pipeline:
                    response = pipeline.followup(followup, result)
                else:
                    pipeline = DeepResearchPipeline(provider=provider)
                    response = pipeline.followup(followup, result)

                st.markdown(response)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )


# ─── Footer ──────────────────────────────────────────

st.markdown("---")
st.caption(
    "Deep Research Agent — Powered by A1, A2, T1, T2 Paradigms | "
    "arXiv + Web Search + PDF Analysis"
)
