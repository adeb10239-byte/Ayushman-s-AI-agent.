import asyncio
import io
import arxiv
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from duckduckgo_search import DDGS

# ReportLab components for high-speed local PDF synthesis
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

# ==========================================
# 1. ARTISTIC THEME & CSS INJECTION SETUP
# ==========================================
st.set_page_config(page_title="Ayushman's AI AGENT", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #15102a 50%, #060210 100%);
        color: #e0d5ff;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(18, 12, 38, 0.7) !important;
        border-right: 1px solid rgba(147, 51, 234, 0.3);
    }
    .neon-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(45deg, #a855f7, #3b82f6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 10px;
        filter: drop-shadow(0px 2px 8px rgba(168, 85, 247, 0.4));
    }
    .neon-subtitle {
        text-align: center;
        color: #93c5fd;
        margin-bottom: 2rem;
    }
    .stChatInputContainer {
        border-radius: 20px !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(45deg, #a855f7, #3b82f6) !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.6) !important;
    }
    div.inline-download-box div[data-testid="stDownloadButton"] > button {
        background: rgba(147, 51, 234, 0.1) !important;
        color: #a855f7 !important;
        border: 1px solid rgba(147, 51, 234, 0.4) !important;
        border-radius: 20px !important;
        padding: 4px 16px !important;
        font-size: 0.85rem !important;
        transition: all 0.3s ease !important;
    }
    div.inline-download-box div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(45deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border-color: transparent !important;
        box-shadow: 0 0 12px rgba(147, 51, 234, 0.5) !important;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = list()

# ==========================================
# 2. STRUCTURAL DATA VALIDATION SCHEMA
# ==========================================
class QueryCorrection(BaseModel):
    corrected_query: str = Field(
        description="The clean, grammatically corrected query string in standard English."
    )

# ==========================================
# 3. NATIVE PDF GENERATION HELPER
# ==========================================
def generate_pdf_bytes(report_text, topic_title):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=18, leading=22, textColor='#1e1b4b',
        spaceAfter=15, alignment=TA_LEFT
    )
    body_style = ParagraphStyle(
        'DocBody', parent=styles['BodyText'],
        fontSize=10, leading=15, textColor='#334155',
        spaceAfter=8, alignment=TA_LEFT
    )
    story = list()
    story.append(Paragraph(f"Deep Research Briefing: {topic_title}", title_style))
    story.append(Spacer(1, 12))
    
    report_content = report_text
    target_header = "📋 Extensive Summary Report"
    
    if target_header in report_text:
        parts = report_text.split(target_header, 1)
        report_content = parts[1].strip()
        story.append(Paragraph(target_header, ParagraphStyle('H4', parent=styles['Heading4'], fontSize=12, textColor='#7c3aed', spaceAfter=10)))
    
    lines = report_content.split('\n')
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            if any(phrase in cleaned_line.lower() for phrase in ["hope this helps", "let me know if you need", "signing off", "have a great day"]):
                continue
            cleaned_line = cleaned_line.replace('**', '').replace('###', '').replace('#', '').strip()
            story.append(Paragraph(cleaned_line, body_style))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 4. LIGHTNING-FAST DATA RETRIEVAL TOOLS
# ==========================================
async def async_web_search(query: str) -> str:
    try:
        loop = asyncio.get_running_loop()
        def fetch_results():
            with DDGS() as ddgs:
                return [r['body'] for r in ddgs.text(query, max_results=2)]
        results = await loop.run_in_executor(None, fetch_results)
        return "\n\n".join(results) if results else "No general web results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"

async def async_arxiv_research(query: str) -> str:
    try:
        loop = asyncio.get_running_loop()
        def fetch_papers():
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=2, sort_by=arxiv.SortCriterion.Relevance)
            papers_data = list()
            for result in client.results(search):
                authors = ", ".join([author.name for author in result.authors])
                paper_brief = (
                    f"🔬 Title: {result.title}\n"
                    f"📅 Published: {result.published.strftime('%Y-%m-%d')}\n"
                    f"✍️ Authors: {authors}\n"
                    f"📝 Abstract Summary: {result.summary[:600]}...\n"
                    f"🔗 Full Document Link: {result.pdf_url}"
                )
                papers_data.append(paper_brief)
            return "\n\n".join(papers_data) if papers_data else "No academic papers discovered on this topic."
        return await loop.run_in_executor(None, fetch_papers)
    except Exception as e:
        return f"Academic retrieval skipped: {str(e)}"

async def auto_correct_prompt(client: genai.Client, raw_prompt: str) -> str:
    system_instruction = (
        "You are an automated query spell-checker. Fix any obvious typos, spelling errors, "
        "or missing letters in the input to make it standard English. Do NOT answer the prompt. "
        "Do NOT explain the topic. Keep the output strictly focused on fixing typographical structural faults.\n\n"
        "EXAMPLES OF CORRECT BEHAVIOR:\n"
        "Input: 'tell me about quantum computin'\n"
        "Output: 'tell me about quantum computing'\n\n"
        "CRITICAL: Do NOT write details. Just fix typos. If the input has no errors, repeat it exactly."
    )
    try:
        response = await client.aio.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=f"Text to fix: {raw_prompt}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=QueryCorrection,
            )
        )
        structured_data = QueryCorrection.model_validate_json(response.text)
        return structured_data.corrected_query.strip()
    except Exception:
        return raw_prompt

async def try_model_generation(client: genai.Client, model_name: str, contents_payload: list, system_instruction: str):
    response = await client.aio.models.generate_content(
        model=model_name, contents=contents_payload, 
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.4)
    )
    return response.text, model_name
# ==========================================
# 5. CORE RUNTIME OPTIMIZATION PIPELINE
# ==========================================
async def run_pipeline(api_key: str, user_prompt: str, agent_role: str, agent_goal: str, enable_web_search: bool, history: list, progress_bar):
    client = genai.Client(api_key=api_key)
    corrected_prompt = user_prompt
    context_data = ""
    academic_data = ""
    
    progress_bar.progress(15)
    
    tasks = [auto_correct_prompt(client, user_prompt)]
    if enable_web_search:
        tasks.append(async_web_search(user_prompt))
        tasks.append(async_arxiv_research(user_prompt))
    
    progress_bar.progress(35)
    completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
    
    if not isinstance(completed_tasks[0], Exception):
        corrected_prompt = completed_tasks[0]
        
    if enable_web_search:
        if not isinstance(completed_tasks[1], Exception):
            context_data = completed_tasks[1]
        if not isinstance(completed_tasks[2], Exception):
            academic_data = completed_tasks[2]
            
    progress_bar.progress(65)
    has_typo = corrected_prompt.lower() != user_prompt.lower()
    system_instruction = f"Role: {agent_role}\nGoal: {agent_goal}\n\nCORE INTERACTION RULE: Act as a brilliant, supportive, human coworker. You are highly analytical and deeply descriptive. Provide highly extensive, thorough explanations without dropping your fast, friendly execution style."
    
    contents_payload = list()
    recent_history = history[-4:] if history else list()
    for turn in recent_history:
        sdk_role = "user" if turn["role"] == "user" else "model"
        contents_payload.append(types.Content(role=sdk_role, parts=[types.Part.from_text(text=turn["content"])]))
    
    final_user_turn = (
        f"User Request: {corrected_prompt}\n\n"
        f"Live Web Data Context:\n{context_data}\n\n"
        f"Verified Academic Whitepapers (ArXiv Database):\n{academic_data}"
    )
    contents_payload.append(types.Content(role="user", parts=[types.Part.from_text(text=final_user_turn)]))
    
    progress_bar.progress(85)
    
    final_prompt = """
    You are performing complete, deep structural analysis context research loops.
    
    Provide your response following this exact comprehensive layout:
    1. A warm, human-like friendly opening addressing the user's focus framework.
    2. A distinct section titled "📋 Extensive Summary Report". Under this header, you MUST structure your findings into these five extensive, paragraph-driven technical sections:
       - 🔍 A. Deep Core Concepts & Historical Context (Incorporate insights from the attached Academic papers if available)
       - ⚙️ B. Technological Infrastructure & Core Workflows
       - 🌍 C. Key Global Impact, Constraints & Limitations
       - 🚀 D. Real-World Case Studies & Contemporary Implementations (Cite specific authors/papers from the academic context if relevant)
       - 🔮 E. Future Horizons & Strategic Research Projections
    3. A brief, encouraging friendly closing sign-off statement.
    """
    contents_payload.append(types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)]))
    generation_tasks = [
        asyncio.create_task(try_model_generation(client, "gemini-3.8-flash", contents_payload, system_instruction)),
        asyncio.create_task(try_model_generation(client, "gemini-3.6-flash", contents_payload, system_instruction))
    ]
    
    # Interrupt processing the moment the absolute fastest model registers its response
    done, pending = await asyncio.wait(generation_tasks, return_when=asyncio.FIRST_COMPLETED)
    
    # Instantly cancel the lagging slow task to minimize device execution queues
    for task in pending:
        task.cancel()
        
    for task in done:
        try:
            text_result, successful_model = task.result()
            progress_bar.progress(100)
            return text_result, successful_model, corrected_prompt, has_typo
        except Exception:
            continue
            
    # Emergency fallback pipeline loop if both flash server clusters face congestion limits
    try:
        text_result, successful_model = await try_model_generation(client, "gemini-3.5-flash-lite", contents_payload, system_instruction)
        progress_bar.progress(100)
        return text_result, successful_model, corrected_prompt, has_typo
    except Exception:
        raise Exception("All system server pipelines are congested. Please try again.")
# ==========================================
# 6. INTERFACE RENDERING & WORKFLOW LOOP
# ==========================================
st.markdown('<div class="neon-title">🤖 AI Agent For Extensive Research on any Topic (By Ayushman Deb)</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-subtitle">Type What\'s on your mind below to get extensive details about it!</div>', unsafe_allow_html=True)

st.sidebar.header("🔧 Configure Your Agent")
api_key = st.sidebar.text_input("Enter Key:", type="password", placeholder="Leave blank to use system key")
agent_role = st.sidebar.text_input("Agent Role:", value="Expert Research Assistant")
agent_goal = st.sidebar.text_area("Agent Goal:", value="Find the latest trends on a given topic.")
agent_backstory = st.sidebar.text_area("Agent Backstory:", value="You are a fast, high-efficiency assistant.")

enable_web_search = st.sidebar.checkbox("Give Agent Web Search Capabilities?", value=True)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = list()
    st.rerun()

# Render chat history with filtered inline PDF download hooks
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            pdf_data = generate_pdf_bytes(message["content"], f"Report Item {idx}")
            st.markdown('<div class="inline-download-box">', unsafe_allow_html=True)
            st.download_button(
                label="📥 Save as PDF",
                data=pdf_data,
                file_name=f"comprehensive_report_{idx}.pdf",
                mime="application/pdf",
                key=f"dl_{idx}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

if user_prompt := st.chat_input("What shall we talk about or research next?"):
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # MULTI-KEY FALLBACK ROUTER: 1. Sidebar input -> 2. Primary Secret -> 3. Backup Secret
    if api_key.strip():
        active_key = api_key
    else:
        active_key = st.secrets.get("GEMINI_API_KEY", st.secrets.get("BACKUP_API_KEY", ""))
        
    if not active_key:
        st.error("Please enter your Key in the sidebar or set up an app Secret to proceed.")
    else:
        with st.chat_message("assistant"):
            p_bar = st.progress(0)
            try:
                compiled_report, successful_model, clean_prompt, fixed = asyncio.run(run_pipeline(
                    api_key=active_key, 
                    user_prompt=user_prompt,
                    agent_role=agent_role,
                    agent_goal=agent_goal,
                    enable_web_search=enable_web_search,
                    history=st.session_state.chat_history,
                    progress_bar=p_bar
                ))
                
                p_bar.empty()
                
                if fixed:
                    st.caption(f"🔍 Auto-Corrected: \"{clean_prompt}\"")
                
                st.markdown(compiled_report)
                
                st.session_state.chat_history.append({"role": "user", "content": user_prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": compiled_report})
                
                current_idx = len(st.session_state.chat_history) - 1
                live_pdf = generate_pdf_bytes(compiled_report, clean_prompt)
                
                st.markdown('<div class="inline-download-box">', unsafe_allow_html=True)
                st.download_button(
                    label="📥 Save as PDF",
                    data=live_pdf,
                    file_name=f"comprehensive_report_{current_idx}.pdf",
                    mime="application/pdf",
                    key=f"dl_live_{current_idx}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                p_bar.empty()
                if "exhausted" in str(e).lower() or "quota" in str(e).lower():
                    st.error("⚠️ System limits reached. Please enter your personal key in the sidebar!")
                else:
                    st.error(f"An error occurred: {str(e)}")
