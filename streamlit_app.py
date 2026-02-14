"""
Streamlit UI for Customer Support Chatbot
==========================================

Interactive chat interface with real-time intent detection and sentiment analysis.
"""

import streamlit as st
import requests
import time
import html
from datetime import datetime
from typing import Optional, Dict

# Configuration
API_URL = "http://localhost:8000"
PAGE_TITLE = "🤖 Customer Support Chatbot"
PAGE_ICON = "🤖"

# Initialize Streamlit page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #1a1033 100%);
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background: rgba(15, 23, 42, 0.92);
        color: #e5e7eb;
        padding: 12px 16px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 70%;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    }
    
    .message-meta {
        font-size: 0.8em;
        opacity: 0.85;
        margin-top: 5px;
        color: #cbd5e1;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(15, 23, 42, 0.92);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi! I'm your AI customer support assistant. How can I help you today?",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"streamlit-{int(time.time())}"
    
    if 'api_status' not in st.session_state:
        st.session_state.api_status = check_api_health()
    
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0
    
    if 'sentiment_stats' not in st.session_state:
        st.session_state.sentiment_stats = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}


def check_api_health() -> Dict:
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy"}
    except:
        return {"status": "unreachable"}


@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_api_stats() -> Optional[Dict]:
    """Get chatbot statistics from API (cached)"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_supported_intents() -> Optional[Dict]:
    """Get list of supported intents (cached)"""
    try:
        response = requests.get(f"{API_URL}/intents", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def send_message(user_message: str) -> Optional[Dict]:
    """Send message to chatbot API"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": user_message,
                "session_id": st.session_state.session_id
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timeout. Try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure it's running.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment"""
    emoji_map = {
        "POSITIVE": "😊",
        "NEGATIVE": "😟",
        "NEUTRAL": "😐"
    }
    return emoji_map.get(sentiment, "😐")


def get_bucket_color(bucket: str) -> str:
    """Get color for bucket badge"""
    color_map = {
        "BUCKET_A": "🟢",
        "BUCKET_B": "🟡",
        "BUCKET_C": "🔴"
    }
    return color_map.get(bucket, "⚪")


def stream_text(text: str, delay: float = 0.01):
    """Yield text in small chunks to simulate streaming output"""
    for token in text.split():
        yield token + " "
        time.sleep(delay)


def render_streaming_bubble(content: str, timestamp: str) -> str:
    """Render streaming bubble HTML for incremental updates"""
    safe_content = html.escape(content)
    return f"""
    <div style='margin: 10px 0;'>
        <div style='background: rgba(15, 23, 42, 0.92); color: #e5e7eb; 
                    padding: 12px 16px; border-radius: 18px; max-width: 75%;
                    border: 1px solid rgba(148, 163, 184, 0.18);
                    box-shadow: 0 2px 10px rgba(0,0,0,0.35);'>
            <div style='word-wrap: break-word; white-space: pre-wrap; margin-bottom: 8px;'>{safe_content}</div>
            <div style='font-size: 0.75em; opacity: 0.8; margin-top: 8px;'>{timestamp}</div>
        </div>
    </div>
    """


def render_sidebar():
    """Render sidebar with stats and controls"""
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        
        # API Status
        status = st.session_state.api_status
        if status.get("status") == "healthy":
            st.success("✅ API Connected")
        elif status.get("status") == "unhealthy":
            st.warning("⚠️ API Unhealthy")
        else:
            st.error("❌ API Unreachable")
        
        st.markdown("---")
        
        # Session Info
        st.markdown("### 💬 Session Info")
        st.metric("Total Messages", st.session_state.total_messages)
        st.caption(f"Session ID: `{st.session_state.session_id[:12]}...`")
        
        st.markdown("---")
        
        # Sentiment Distribution
        st.markdown("### 🎭 Sentiment Distribution")
        sentiment_stats = st.session_state.sentiment_stats
        total = sum(sentiment_stats.values())
        
        if total > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                pct = (sentiment_stats["POSITIVE"] / total) * 100
                st.metric("😊", f"{pct:.0f}%")
            with col2:
                pct = (sentiment_stats["NEUTRAL"] / total) * 100
                st.metric("😐", f"{pct:.0f}%")
            with col3:
                pct = (sentiment_stats["NEGATIVE"] / total) * 100
                st.metric("😟", f"{pct:.0f}%")
        else:
            st.info("No messages yet")
        
        st.markdown("---")
        
        # Chatbot Stats
        st.markdown("### 🤖 Chatbot Stats")
        stats = get_api_stats()
        
        if stats:
            st.metric("Model Accuracy", stats.get("model_accuracy", "N/A"))
            st.metric("Total Intents", stats.get("total_intents", "N/A"))
            st.metric("Cost Savings", stats.get("cost_savings", "N/A"))
            
            with st.expander("📦 Bucket Distribution"):
                dist = stats.get("routing_distribution", {})
                st.write(f"🟢 Bucket A: {dist.get('BUCKET_A', 'N/A')}")
                st.write(f"🟡 Bucket B: {dist.get('BUCKET_B', 'N/A')}")
                st.write(f"🔴 Bucket C: {dist.get('BUCKET_C', 'N/A')}")
        else:
            st.info("Stats unavailable")
        
        st.markdown("---")
        
        # Supported Intents
        with st.expander("🎯 Supported Intents"):
            intents = get_supported_intents()
            if intents:
                buckets = intents.get("buckets", {})
                for bucket_name, bucket_data in buckets.items():
                    st.markdown(f"**{get_bucket_color(bucket_name)} {bucket_name}**")
                    st.caption(bucket_data.get("description", ""))
                    with st.container():
                        for intent in bucket_data.get("intents", [])[:5]:
                            st.text(f"• {intent}")
                        remaining = len(bucket_data.get("intents", [])) - 5
                        if remaining > 0:
                            st.caption(f"... and {remaining} more")
            else:
                st.info("Intents unavailable")
        
        st.markdown("---")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hi! I'm your AI customer support assistant. How can I help you today?",
                    "timestamp": datetime.now().strftime("%H:%M")
                }
            ]
            st.session_state.total_messages = 0
            st.session_state.sentiment_stats = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
            st.rerun()
        
        # Refresh API Status
        if st.button("🔄 Refresh API Status", use_container_width=True):
            st.session_state.api_status = check_api_health()
            st.rerun()


def render_chat_message(message: Dict):
    """Render a single chat message"""
    role = message["role"]
    content = message["content"]
    timestamp = message.get("timestamp", "")
    
    # Escape HTML special characters in content
    safe_content = html.escape(content)
    
    if role == "user":
        st.markdown(f"""
        <div style='text-align: right; margin: 10px 0;'>
            <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 12px 16px; border-radius: 18px; max-width: 75%; text-align: left;'>
                <div style='word-wrap: break-word; white-space: pre-wrap;'>{safe_content}</div>
                <div style='font-size: 0.75em; opacity: 0.8; margin-top: 5px;'>{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='margin: 10px 0;'>
            <div style='background: rgba(15, 23, 42, 0.92); color: #e5e7eb; 
                        padding: 12px 16px; border-radius: 18px; max-width: 75%;
                        border: 1px solid rgba(148, 163, 184, 0.18);
                        box-shadow: 0 2px 10px rgba(0,0,0,0.35);'>
                <div style='word-wrap: break-word; white-space: pre-wrap; margin-bottom: 8px;'>{safe_content}</div>
                <div style='font-size: 0.75em; opacity: 0.8; margin-top: 8px;'>{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render metadata as plain captions to avoid raw HTML output
        if "metadata" in message:
            meta = message["metadata"]
            sentiment_emoji = get_sentiment_emoji(meta.get("sentiment", "NEUTRAL"))
            bucket_emoji = get_bucket_color(meta.get("bucket", ""))
            escalated = "⚠️ ESCALATED" if meta.get("escalated_by_sentiment") else ""
            latency_info = f"⏱️ {meta.get('latency_ms', 0):.0f}ms" if 'latency_ms' in meta else ""
            cost_usd = meta.get("cost_usd", None)
            cost_info = f"💵 ${cost_usd:.6f}" if isinstance(cost_usd, (int, float)) else ""

            st.caption(f"🎯 {meta.get('intent', 'unknown')} ({int(meta.get('confidence', 0) * 100)}% confidence)")
            st.caption(
                f"{bucket_emoji} {meta.get('bucket', 'N/A')} • {meta.get('cost_tier', 'N/A')} cost"
                f"{' • ' + cost_info if cost_info else ''}"
            )
            st.caption(
                f"{sentiment_emoji} {meta.get('sentiment', 'N/A')}{' • ' + latency_info if latency_info else ''}{' • ' + escalated if escalated else ''}"
            )


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 Customer Support Chatbot</h1>
        <p>Powered by AI • 79.6% cost savings • 27 intents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display all messages
        for message in st.session_state.messages:
            render_chat_message(message)
    
    # Chat input at the bottom
    st.markdown("---")
    
    # Check API status before allowing input
    if st.session_state.api_status.get("status") != "healthy":
        st.warning("⚠️ API is not available. Please check the API status in the sidebar.")
        return
    
    # User input
    user_input = st.chat_input("Type your message here...", key="user_input")
    
    if user_input:
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(user_message)
        st.session_state.total_messages += 1
        
        # Show typing indicator
        with st.spinner("🤖 Thinking..."):
            # Send to API
            response_data = send_message(user_input)
        
        if response_data:
            # Update sentiment stats
            sentiment = response_data.get("sentiment", "NEUTRAL")
            if sentiment in st.session_state.sentiment_stats:
                st.session_state.sentiment_stats[sentiment] += 1

            # Stream the assistant response before committing to history
            stream_timestamp = datetime.now().strftime("%H:%M")
            stream_placeholder = st.empty()
            streamed_text = ""
            for chunk in stream_text(response_data.get("response", "")):
                streamed_text += chunk
                stream_placeholder.markdown(
                    render_streaming_bubble(streamed_text, stream_timestamp),
                    unsafe_allow_html=True
                )
            stream_placeholder.empty()
            
            # Add bot response
            bot_message = {
                "role": "assistant",
                "content": response_data.get("response", "Sorry, I couldn't process that."),
                "timestamp": stream_timestamp,
                "metadata": {
                    "intent": response_data.get("intent"),
                    "confidence": response_data.get("confidence"),
                    "bucket": response_data.get("bucket"),
                    "cost_tier": response_data.get("cost_tier"),
                    "cost_usd": response_data.get("cost_usd", 0.0),
                    "sentiment": response_data.get("sentiment"),
                    "sentiment_score": response_data.get("sentiment_score"),
                    "escalated_by_sentiment": response_data.get("escalated_by_sentiment"),
                    "latency_ms": response_data.get("latency_ms", 0)
                }
            }
            st.session_state.messages.append(bot_message)
        
        # Rerun to update UI
        st.rerun()


if __name__ == "__main__":
    main()
