import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from emotion_detector import NaiveBayesEmotionClassifier, KeywordEmotionAnalyzer, get_mixed_emotions
from groq_integration import get_groq_response, EMOTION_RESPONSES

# ─── 1. Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── 2. Custom CSS for Premium Look ─────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }

    /* Emotion result cards */
    .emotion-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .emotion-card .emoji {
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }
    .emotion-card .label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .emotion-card .conf {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* AI Response box */
    .ai-response-box {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1), rgba(59, 130, 246, 0.05));
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Smooth animations */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── 3. Session State ────────────────────────────────────────────────────────
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

CSV_FILE = 'history.csv'


def save_to_csv(record):
    df_new = pd.DataFrame([record])
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(CSV_FILE, index=False)
    else:
        df_new.to_csv(CSV_FILE, index=False)


def get_csv_count():
    if os.path.exists(CSV_FILE):
        return len(pd.read_csv(CSV_FILE))
    return 0


# ─── 4. Cache Models ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    nb_model = NaiveBayesEmotionClassifier()
    kw_model = KeywordEmotionAnalyzer()
    return nb_model, kw_model


nb_classifier, kw_analyzer = load_models()

# ─── 5. Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    st.markdown('<span class="status-badge badge-success">✅ Models Ready</span>', unsafe_allow_html=True)

    st.markdown("---")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Sessions", len(st.session_state.emotion_history))
    with col_s2:
        st.metric("CSV Records", get_csv_count())

    st.markdown("---")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.emotion_history = []
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        st.rerun()

    if st.session_state.emotion_history:
        st.markdown("### 🕐 Recent")
        for item in reversed(st.session_state.emotion_history[-3:]):
            emoji = EMOTION_RESPONSES.get(item['emotion'], {}).get('emoji', '❓')
            st.markdown(f"**{emoji} {item['emotion']}** — {item['field']}  \n`{item['confidence']:.0%} confidence`")

# ─── 6. Main Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Learning Assistant</h1>
    <p>Emotion-aware guidance powered by AI — Get personalized help based on how you feel</p>
</div>
""", unsafe_allow_html=True)

# ─── 7. Input Section ────────────────────────────────────────────────────────
col_input, col_settings = st.columns([2.5, 1])

with col_input:
    st.markdown("### 📚 Describe Your Learning Challenge")

    field_options = [
        "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
        "Engineering", "Business", "Literature", "History", "Psychology", "Other"
    ]

    field = st.selectbox(
        "What field are you studying?",
        field_options,
        help="Select your area of study for personalized responses"
    )

    problem = st.text_area(
        f"What's your {field} challenge?",
        placeholder=f"e.g., 'I'm struggling with recursion' or 'This concept is so confusing I don't know where to start'",
        height=120
    )

    # Quick example buttons
    st.markdown("**⚡ Quick Examples:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("😕 I'm confused"):
            problem = "I'm confused about recursion and don't understand how it works"
    with qc2:
        if st.button("😤 So frustrated"):
            problem = "Debugging is so frustrating, nothing works and I keep getting errors"
    with qc3:
        if st.button("🤩 I'm curious"):
            problem = "I'm curious about machine learning, it's fascinating and I want to learn more"
    with qc4:
        if st.button("🥱 So bored"):
            problem = "This topic is boring and tedious, I can't stay focused"

    submit = st.button("🔍 Analyze & Get AI Help", type="primary", use_container_width=True)

with col_settings:
    st.markdown("### ⚙️ Settings")

    use_ai = st.checkbox("Use AI Response (Groq)", value=True,
                         help="Generate personalized responses using Groq AI")
    save_data = st.checkbox("Save to CSV", value=True,
                            help="Save interactions for learning analytics")
    show_details = st.checkbox("Show analysis details", value=True)

    st.markdown("---")
    st.markdown("### 📊 Data Source")
    use_csv = st.checkbox("Use CSV history for charts", value=False)
    if use_csv:
        count = get_csv_count()
        if count > 0:
            st.info(f"📁 {count} saved records")
        else:
            st.warning("No CSV data yet")

# ─── 8. Analysis & Results ───────────────────────────────────────────────────
if submit and problem:
    with st.spinner("🔬 Analyzing your emotional state..."):

        # Run both classifiers
        nb_result = nb_classifier.predict(problem)
        kw_result = kw_analyzer.predict(problem)

    # ── Model Comparison ──
    st.markdown("---")
    st.markdown("### 🔬 Emotion Analysis Results")

    res_col1, res_col2 = st.columns(2)

    # NB Classifier Column
    with res_col1:
        st.markdown("#### 🤖 ML Classifier (Naive Bayes)")
        nb_mixed = get_mixed_emotions(nb_result['scores'])

        em = nb_result['emotion']
        emoji = EMOTION_RESPONSES.get(em, {}).get("emoji", "❓")
        conf = nb_result['confidence']

        st.markdown(f"""
        <div class="emotion-card">
            <div class="emoji">{emoji}</div>
            <div class="label">{em}</div>
            <div class="conf">{conf:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

        for emotion_name, score in sorted(nb_result['scores'].items(), key=lambda x: x[1], reverse=True):
            e = EMOTION_RESPONSES.get(emotion_name, {}).get("emoji", "")
            st.progress(float(score), text=f"{e} {emotion_name}: {score:.1%}")

    # Keyword Analyzer Column
    with res_col2:
        st.markdown("#### 🔑 Keyword Analyzer")
        kw_mixed = get_mixed_emotions(kw_result['scores'])

        em2 = kw_result['emotion']
        emoji2 = EMOTION_RESPONSES.get(em2, {}).get("emoji", "❓")
        conf2 = kw_result['confidence']

        st.markdown(f"""
        <div class="emotion-card">
            <div class="emoji">{emoji2}</div>
            <div class="label">{em2}</div>
            <div class="conf">{conf2:.1%} confidence</div>
        </div>
        """, unsafe_allow_html=True)

        for emotion_name, score in sorted(kw_result['scores'].items(), key=lambda x: x[1], reverse=True):
            e = EMOTION_RESPONSES.get(emotion_name, {}).get("emoji", "")
            st.progress(float(score), text=f"{e} {emotion_name}: {score:.1%}")

    # ── Agreement indicator ──
    if nb_result['emotion'] == kw_result['emotion']:
        st.success(f"✅ Both models agree: **{nb_result['emotion']}** — High confidence in this prediction!")
    else:
        st.warning(f"⚠️ Models differ: ML says **{nb_result['emotion']}**, Keywords say **{kw_result['emotion']}** — Using ML prediction as primary.")

    # ── AI Response ──
    st.markdown("---")
    st.markdown("### 🤖 AI Learning Assistant Response")

    primary_emotion = nb_result['emotion']
    primary_confidence = nb_result['confidence']

    primary_emoji = EMOTION_RESPONSES.get(primary_emotion, {}).get("emoji", "")
    st.markdown(f"💡 Responding to detected emotion: **{primary_emoji} {primary_emotion}** ({primary_confidence:.0%} confidence)")

    if use_ai:
        with st.spinner("✨ Generating personalized AI response..."):
            ai_response = get_groq_response(field, problem, primary_emotion, primary_confidence)
    else:
        ai_response = EMOTION_RESPONSES.get(primary_emotion, {}).get("message", "Keep going!")

    st.markdown(f"""
    <div class="ai-response-box">
        {ai_response}
    </div>
    """, unsafe_allow_html=True)

    # ── Details Expander ──
    if show_details:
        with st.expander("📋 Analysis Details"):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"**Original Input:** {problem}")
                st.markdown(f"**Processed Text:** {nb_result['cleaned_text']}")
                st.markdown(f"**Primary Emotion:** {primary_emotion}")
            with detail_col2:
                st.markdown(f"**ML Confidence:** {nb_result['confidence']:.3f}")
                st.markdown(f"**Keyword Confidence:** {kw_result['confidence']:.3f}")
                st.markdown(f"**AI Model:** Llama 3.1 (via Groq)")
                st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Save Record ──
    record = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "field": field,
        "problem": problem,
        "emotion": primary_emotion,
        "confidence": round(primary_confidence, 4),
        "model": "NaiveBayes"
    }

    st.session_state.emotion_history.append(record)
    if save_data:
        save_to_csv(record)
        # Also save keyword analyzer result
        kw_record = record.copy()
        kw_record["emotion"] = kw_result['emotion']
        kw_record["confidence"] = round(kw_result['confidence'], 4)
        kw_record["model"] = "Keyword"
        save_to_csv(kw_record)

# ─── 9. Analytics Dashboard ──────────────────────────────────────────────────
if st.session_state.emotion_history or get_csv_count() > 0:
    st.markdown("---")
    st.markdown("### 📈 Learning Analytics")

    # Load data
    if use_csv and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(st.session_state.emotion_history)

    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        tab1, tab2, tab3 = st.tabs(["🎭 Emotions", "📚 Fields", "📋 Summary"])

        with tab1:
            chart_c1, chart_c2 = st.columns(2)
            with chart_c1:
                emotion_counts = df['emotion'].value_counts()

                # Custom color map
                color_map = {
                    "Bored": "#94A3B8",
                    "Confident": "#22C55E",
                    "Confused": "#F59E0B",
                    "Curious": "#3B82F6",
                    "Frustrated": "#EF4444"
                }
                colors = [color_map.get(e, "#8B5CF6") for e in emotion_counts.index]

                fig1 = go.Figure(data=[go.Pie(
                    labels=emotion_counts.index,
                    values=emotion_counts.values,
                    hole=0.4,
                    marker_colors=colors,
                    textinfo='label+percent',
                    textfont_size=13
                )])
                fig1.update_layout(
                    title="Emotion Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#E2E8F0',
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig1, use_container_width=True)

            with chart_c2:
                df_copy = df.copy()
                df_copy['time'] = df_copy['timestamp'].dt.strftime('%H:%M:%S')
                fig2 = px.line(
                    df_copy, x='time', y='confidence', color='emotion',
                    title="Emotional Journey",
                    markers=True,
                    color_discrete_map=color_map
                )
                fig2.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#E2E8F0',
                    height=350
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            if 'model' in df.columns:
                field_emotion = df.groupby(['field', 'emotion', 'model']).size().reset_index(name='count')
                fig3 = px.bar(
                    field_emotion, x='field', y='count', color='emotion',
                    facet_col='model',
                    title="Emotions by Study Field & Model",
                    color_discrete_map=color_map
                )
            else:
                field_emotion = df.groupby(['field', 'emotion']).size().reset_index(name='count')
                fig3 = px.bar(
                    field_emotion, x='field', y='count', color='emotion',
                    title="Emotions by Study Field",
                    color_discrete_map=color_map
                )
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#E2E8F0',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            st.markdown("#### Recent Activity")
            st.dataframe(
                df.tail(10).style.format({'confidence': '{:.1%}'}),
                use_container_width=True
            )

# ─── 10. Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "🧠 AI Learning Assistant — Built with Streamlit, scikit-learn & Groq AI"
    "</p>",
    unsafe_allow_html=True
)
