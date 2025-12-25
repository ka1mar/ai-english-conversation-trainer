from agno.agent import Agent
from agno.models.deepseek import DeepSeek
import streamlit as st
from transformers import pipeline
import tempfile
import os
import logging
from pathlib import Path
import soundfile as sf
import torch
from kokoro import KPipeline
import base64
from embedchain import App
import time
import warnings
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import re
import numpy as np

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

_background_cache = {
    "weak_areas": None,
    "last_updated": 0
}

@st.cache_resource
def load_whisper_model():
    """Load distil-whisper model - дистиллированная версия в 6x быстрее!"""
    try:
        start_time = time.time()
        print("⏳ Загрузка distil-whisper модели (оптимизированная версия)...")
        
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/distil-small.en",
            device=-1,
            torch_dtype=torch.float32
        )
        
        elapsed = time.time() - start_time
        print(f"✅ distil-whisper модель загружена за {elapsed:.2f} секунд (уже в 6x быстрее!)")
        return whisper_pipeline
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        print(f"❌ Ошибка загрузки Whisper: {str(e)}")
        return None

@st.cache_resource
def load_tts_model():
    """Load Kokoro TTS model"""
    try:
        start_time = time.time()
        print("⏳ Загрузка TTS модели...")
        tts_pipeline = KPipeline(lang_code='a')
        elapsed = time.time() - start_time
        print(f"✅ TTS модель загружена за {elapsed:.2f} секунд")
        return tts_pipeline
    except Exception as e:
        logger.error(f"Error loading TTS model: {str(e)}")
        return None

def transcribe_audio(audio_file, whisper_pipeline):
    """Transcribe audio file using distil-whisper"""
    try:
        start_time = time.time()
        print("⏳ Начинаем распознавание речи (distil-whisper)...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        result = whisper_pipeline(tmp_path)
        
        os.unlink(tmp_path)
        
        elapsed = time.time() - start_time
        print(f"✅ Распознавание завершено за {elapsed:.2f} секунд")
        
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        print(f"❌ Ошибка распознавания: {str(e)}")
        return None

@st.cache_resource
def create_rag_app(deepseek_key):
    """Create RAG app with embedchain using DeepSeek for LLM and HuggingFace for embeddings"""
    try:
        start_time = time.time()
        print("⏳ Создание RAG приложения...")
        
        db_path = tempfile.mkdtemp()
        app = App.from_config(
            config={
                "llm": {
                    "provider": "openai",
                    "config": {
                        "api_key": deepseek_key,
                        "model": "deepseek-chat",
                        "base_url": "https://api.deepseek.com"
                    }
                },
                "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                },
            }
        )
        elapsed = time.time() - start_time
        print(f"✅ RAG приложение создано за {elapsed:.2f} секунд")
        return app
    except Exception as e:
        logger.error(f"Error creating RAG app: {str(e)}")
        return None

@st.cache_resource
def create_error_memory_app(deepseek_key):
    """Create separate RAG app for storing user's grammar mistakes and weak areas"""
    try:
        start_time = time.time()
        print("⏳ Создание приложения памяти ошибок...")
        
        db_path = tempfile.mkdtemp()
        app = App.from_config(
            config={
                "llm": {
                    "provider": "openai",
                    "config": {
                        "api_key": deepseek_key,
                        "model": "deepseek-chat",
                        "base_url": "https://api.deepseek.com"
                    }
                },
                "vectordb": {"provider": "chroma", "config": {"dir": db_path, "collection_name": "error_memory"}},
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2"
                    }
                },
            }
        )
        elapsed = time.time() - start_time
        print(f"✅ Приложение памяти ошибок создано за {elapsed:.2f} секунд")
        return app
    except Exception as e:
        logger.error(f"Error creating error memory app: {str(e)}")
        return None

def add_pdf_to_rag(rag_app, pdf_file):
    """Add PDF to RAG knowledge base"""
    try:
        start_time = time.time()
        print("⏳ Добавление PDF в базу знаний...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_file.getvalue())
            temp_path = f.name
        
        rag_app.add(temp_path, data_type="pdf_file")
        os.remove(temp_path)
        
        elapsed = time.time() - start_time
        print(f"✅ PDF добавлен в базу знаний за {elapsed:.2f} секунд")
        return True
    except Exception as e:
        logger.error(f"Error adding PDF to RAG: {str(e)}")
        return False

def query_rag(rag_app, query):
    """Query RAG system for relevant information"""
    try:
        if rag_app:
            start_time = time.time()
            print("⏳ Запрос к RAG системе...")
            
            response = rag_app.query(query)
            
            elapsed = time.time() - start_time
            print(f"✅ RAG запрос выполнен за {elapsed:.2f} секунд")
            return response
        return None
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return None

def initialize_error_analyzer_agent(api_key: str) -> Agent:
    """Initialize agent for analyzing grammar mistakes"""
    try:
        model = DeepSeek(id="deepseek-chat", api_key=api_key)
        
        analyzer_agent = Agent(
            model=model,
            name="Grammar Error Analyzer",
            instructions=[
                "You are an expert English teacher analyzing student speech for learning patterns.",
                "Your task: Identify grammar mistakes and weak areas in the student's English.",
                "",
                "Analyze the student's text and provide:",
                "1. ERRORS: List specific mistakes (tense errors, article usage, prepositions, etc.)",
                "2. WEAK_AREAS: Categorize recurring problems (e.g., 'past tense', 'articles', 'prepositions')",
                "3. SEVERITY: Rate each area as 'high', 'medium', or 'low' priority",
                "",
                "Format your response as:",
                "ERRORS:",
                "- [specific mistake description]",
                "",
                "WEAK_AREAS:",
                "- [area]: [severity] - [brief explanation]",
                "",
                "Example:",
                "ERRORS:",
                "- Used 'I go' instead of 'I went' (past tense error)",
                "- Missing article: 'I saw dog' should be 'I saw a dog'",
                "",
                "WEAK_AREAS:",
                "- past_tense: high - Multiple errors with past simple tense",
                "- articles: medium - Occasional missing articles",
                "",
                "If the text is perfect, respond with 'NO_ERRORS'."
            ],
            markdown=False
        )
        
        return analyzer_agent
    except Exception as e:
        logger.error(f"Error initializing error analyzer: {str(e)}")
        return None

def analyze_user_errors(user_text: str, api_key: str, error_memory_app):
    """Analyze user's text for errors and store in memory"""
    try:
        start_time = time.time()
        print("⏳ Анализ ошибок пользователя...")
        
        analyzer = initialize_error_analyzer_agent(api_key)
        if not analyzer:
            return None
        
        analysis = analyzer.run(f"Analyze this student text: '{user_text}'")
        
        if analysis and analysis.content:
            if "NO_ERRORS" not in analysis.content.upper():
                error_memory_app.add(
                    f"User text: {user_text}\n\nAnalysis: {analysis.content}",
                    data_type="text"
                )
                elapsed = time.time() - start_time
                print(f"✅ Анализ ошибок завершен за {elapsed:.2f} секунд")
                return analysis.content
        
        elapsed = time.time() - start_time
        print(f"✅ Анализ завершен (ошибок не найдено) за {elapsed:.2f} секунд")
        return None
    except Exception as e:
        logger.error(f"Error analyzing user errors: {str(e)}")
        return None

def get_weak_areas_context(error_memory_app):
    """Query error memory to get summary of weak areas"""
    try:
        if error_memory_app:
            start_time = time.time()
            print("⏳ Получение слабых мест из памяти...")
            
            context = error_memory_app.query(
                "What are the recurring grammar mistakes and weak areas? Summarize the main problems.",
                citations=False
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Слабые места получены за {elapsed:.2f} секунд")
            return context
        return None
    except Exception as e:
        logger.error(f"Error getting weak areas: {str(e)}")
        return None

def split_into_sentences(text):
    """Split text into sentences for parallel TTS processing"""
    
    text = text.strip()
    if not text:
        return []
    
    sentences = []
    current = ""
    
    i = 0
    while i < len(text):
        current += text[i]
        
        if text[i] in '.!?' and i + 1 < len(text):
            if text[i + 1] == ' ':
                if i + 2 < len(text):
                    next_char = text[i + 2]
                    if next_char.isupper() or next_char in '"\'(':
                        sentences.append(current.strip())
                        current = ""
                        i += 2
                        continue
        i += 1
    
    if current.strip():
        sentences.append(current.strip())
    
    return sentences

def generate_tts_for_sentence(args):
    """Generate TTS for a single sentence (used in parallel processing)"""
    sentence, tts_pipeline, voice, sentence_idx = args
    try:
        audio = text_to_speech_raw(sentence, tts_pipeline, voice)
        return (sentence_idx, audio)
    except Exception as e:
        logger.error(f"Error generating TTS for sentence {sentence_idx}: {str(e)}")
        return (sentence_idx, None)

def text_to_speech_raw(text, tts_pipeline, voice='af_heart'):
    """Convert text to speech - returns RAW numpy array (for concatenation)"""
    try:
        start_time = time.time()
        print(f"⏳ Генерация речи из текста...")
        
        generator = tts_pipeline(text, voice=voice)
        audio_chunks = []
        
        for gs, ps, audio in generator:
            audio_chunks.append(audio)
        
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            elapsed = time.time() - start_time
            print(f"✅ Речь сгенерирована за {elapsed:.2f} секунд")
            return full_audio
        
        return None
    except Exception as e:
        logger.error(f"Error in text-to-speech-raw: {str(e)}")
        return None

def text_to_speech(text, tts_pipeline, voice='af_heart'):
    """Convert text to speech using Kokoro TTS - returns file path"""
    try:
        full_audio = text_to_speech_raw(text, tts_pipeline, voice)
        
        if full_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, full_audio, 24000)
                return tmp_file.name
        
        return None
    except Exception as e:
        logger.error(f"Error in text-to-speech: {str(e)}")
        return None

def text_to_speech_raw_parallel(text, tts_pipeline, voice='af_heart', max_workers=3):
    """
    Convert text to speech with PARALLEL sentence processing - returns RAW numpy array.
    
    Splits text into sentences and generates TTS for each in parallel,
    then concatenates the results.
    """
    try:
        start_time = time.time()
        
        sentences = split_into_sentences(text)
        
        if not sentences:
            return text_to_speech_raw(text, tts_pipeline, voice)
        
        if len(sentences) <= 2:
            return text_to_speech_raw(text, tts_pipeline, voice)
        
        print(f"   └── {len(sentences)} предложений - параллельная генерация ({max_workers} потоков)")
        
        args_list = [
            (sentence, tts_pipeline, voice, idx) 
            for idx, sentence in enumerate(sentences)
        ]
        
        audio_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(generate_tts_for_sentence, args): args[3] 
                for args in args_list
            }
            
            for future in as_completed(futures):
                sentence_idx, audio = future.result()
                if audio is not None:
                    audio_results[sentence_idx] = audio
                    print(f"      ✓ Предложение {sentence_idx + 1}/{len(sentences)} готово")
        
        ordered_audio = []
        for idx in range(len(sentences)):
            if idx in audio_results:
                ordered_audio.append(audio_results[idx])
        
        if ordered_audio:
            full_audio = np.concatenate(ordered_audio) if len(ordered_audio) > 1 else ordered_audio[0]
            elapsed = time.time() - start_time
            print(f"✅ Речь сгенерирована за {elapsed:.2f} секунд (параллельно)")
            return full_audio
        
        return None
    except Exception as e:
        logger.error(f"Error in parallel text-to-speech-raw: {str(e)}")
        return text_to_speech_raw(text, tts_pipeline, voice)

def text_to_speech_parallel(text, tts_pipeline, voice='af_heart', max_workers=3):
    """
    Convert text to speech with PARALLEL sentence processing - returns file path.
    
    Splits text into sentences and generates TTS for each in parallel,
    then concatenates the results. This provides significant speedup
    for longer texts.
    
    Flow:
    Sentence 1 ──► TTS Thread 1 ──┐
    Sentence 2 ──► TTS Thread 2 ──┼──► Concatenate ──► Final Audio
    Sentence 3 ──► TTS Thread 3 ──┘
    """
    try:
        start_time = time.time()
        print("⏳ Генерация речи (параллельно по предложениям)...")
        
        full_audio = text_to_speech_raw_parallel(text, tts_pipeline, voice, max_workers)
        
        if full_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, full_audio, 24000)
                return tmp_file.name
        
        return None
    except Exception as e:
        logger.error(f"Error in parallel text-to-speech: {str(e)}")
        return text_to_speech(text, tts_pipeline, voice)

def get_audio_html(audio_path, autoplay=True, show_controls=True):
    """Generate HTML for audio playback"""
    try:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        autoplay_attr = 'autoplay' if autoplay else ''
        controls_attr = 'controls' if show_controls else ''
        
        if autoplay and not show_controls:
            html = f'''
            <audio {autoplay_attr} style="display: none;">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            '''
        else:
            html = f'''
            <audio {autoplay_attr} {controls_attr} style="width: 100%;">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            '''
        return html
    except Exception as e:
        logger.error(f"Error creating audio HTML: {str(e)}")
        return None

def get_streaming_audio_html(audio_chunks_b64, autoplay=True):
    """
    Generate HTML for STREAMING audio playback.
    Plays chunks in sequence - first chunk starts immediately,
    others queue up and play after previous finishes.
    
    audio_chunks_b64: list of base64-encoded audio strings
    """
    if not audio_chunks_b64:
        return None
    
    queue_id = f"audioQueue_{int(time.time() * 1000)}"
    
    chunks_js = ",".join([f'"{chunk}"' for chunk in audio_chunks_b64])
    
    html = f'''
    <div id="{queue_id}_container">
        <audio id="{queue_id}_player" style="display: none;"></audio>
    </div>
    <script>
    (function() {{
        const audioChunks = [{chunks_js}];
        let currentIndex = 0;
        const player = document.getElementById('{queue_id}_player');
        
        function playNext() {{
            if (currentIndex < audioChunks.length) {{
                player.src = 'data:audio/wav;base64,' + audioChunks[currentIndex];
                player.play().catch(e => console.log('Autoplay blocked:', e));
                currentIndex++;
            }}
        }}
        
        player.onended = playNext;
        player.onerror = function() {{
            console.log('Audio error, trying next chunk');
            currentIndex++;
            playNext();
        }};
        
        // Start playing first chunk immediately
        {"playNext();" if autoplay else "// Autoplay disabled"}
    }})();
    </script>
    '''
    return html

def audio_to_base64(audio_data, sample_rate=24000):
    """Convert numpy audio array to base64 string"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
            os.unlink(tmp_file.name)
            return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        logger.error(f"Error converting audio to base64: {str(e)}")
        return None

def initialize_conversation_agent(api_key: str, has_pdf: bool = False, weak_areas: str = None) -> Agent:
    """Initialize conversational AI agent with adaptive practice based on weak areas"""
    try:
        model = DeepSeek(id="deepseek-chat", api_key=api_key)
        
        base_instructions = [
            "You are a friendly English conversation partner helping someone practice English.",
            "Your role:",
            "1. Have a natural, engaging conversation - respond to what they say",
            "2. Keep the conversation going by asking follow-up questions",
            "3. Speak naturally as if in a real conversation",
            "4. Keep responses concise (2-3 sentences) for natural flow",
            "5. Be encouraging and supportive",
        ]
        
        if weak_areas:
            base_instructions.extend([
                "",
                "🎯 ADAPTIVE PRACTICE MODE:",
                "Based on previous conversations, focus on these weak areas:",
                weak_areas,
                "",
                "Your strategy:",
                "- Naturally incorporate practice opportunities for these weak areas",
                "- Ask questions that require using the problematic grammar structures",
                "- Use these structures in your responses as examples",
                "- If past tense is weak: ask about past experiences, what they did yesterday/last week",
                "- If articles are weak: use many examples with articles in your speech",
                "- If prepositions are weak: use phrases with correct prepositions",
                "- Don't explicitly mention you're practicing these areas - keep it natural!",
                ""
            ])
        
        if has_pdf:
            base_instructions.extend([
                "",
                "📄 DOCUMENT CONTEXT:",
                "The student has uploaded a document to discuss.",
                "You will receive relevant excerpts from the document based on the conversation.",
                "Use this information to:",
                "- Discuss topics, ideas, and concepts from the document",
                "- Ask questions about their understanding of the content",
                "- Help them practice vocabulary and phrases from the text",
                "- Encourage them to express opinions about what they read",
                ""
            ])
        
        base_instructions.extend([
            "",
            "Important: Give TWO responses in your output:",
            "1. CONVERSATION: Your natural spoken response (what will be read aloud)",
            "2. FEEDBACK: Brief grammar/vocabulary corrections (shown only in transcript)",
            "",
            "Format:",
            "CONVERSATION: [Your natural response here]",
            "",
            "FEEDBACK: [Quick tips about mistakes, if any. Say 'Great job!' if no mistakes]",
            "",
            "Keep it conversational and natural!"
        ])
        
        conversation_agent = Agent(
            model=model,
            name="English Conversation Partner",
            instructions=base_instructions,
            markdown=False
        )
        
        return conversation_agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def parse_agent_response(response_text):
    """Parse agent response into conversation and feedback parts"""
    try:
        parts = response_text.split("FEEDBACK:")
        conversation = parts[0].replace("CONVERSATION:", "").strip()
        feedback = parts[1].strip() if len(parts) > 1 else "Great job!"
        return conversation, feedback
    except:
        return response_text, "Great job!"

st.set_page_config(
    page_title="Speech-to-Speech English Practice",
    page_icon="🗣️",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = ""

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "played_audio_ids" not in st.session_state:
    st.session_state.played_audio_ids = set()

if "show_feedback_panel" not in st.session_state:
    st.session_state.show_feedback_panel = False

if "rag_app" not in st.session_state:
    st.session_state.rag_app = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "error_memory_app" not in st.session_state:
    st.session_state.error_memory_app = None

if "error_analysis_enabled" not in st.session_state:
    st.session_state.error_analysis_enabled = True

if "total_start_time" not in st.session_state:
    st.session_state.total_start_time = None

if "cached_weak_areas" not in st.session_state:
    st.session_state.cached_weak_areas = None

if "weak_areas_updating" not in st.session_state:
    st.session_state.weak_areas_updating = False

with st.sidebar:
    st.header("🔑 Settings")
    
    api_key = st.text_input("🔑 Enter your DeepSeek API Key:", type="password", value=st.session_state.api_key_input)
    
    if api_key:
        st.session_state.api_key_input = api_key
        st.success("DeepSeek API Key provided! ✅")
    
    st.markdown("---")
    
    st.markdown("### 🧠 Adaptive Learning")
    st.markdown("AI analyzes your mistakes and adapts practice")
    
    if st.session_state.error_analysis_enabled and st.session_state.error_memory_app is None and st.session_state.api_key_input:
        st.session_state.error_memory_app = create_error_memory_app(st.session_state.api_key_input)
        
        if st.session_state.error_memory_app and st.session_state.cached_weak_areas is None:
            def prefetch_weak_areas(memory_app):
                try:
                    return get_weak_areas_context(memory_app)
                except:
                    return None
            
            try:
                st.session_state.cached_weak_areas = prefetch_weak_areas(st.session_state.error_memory_app)
                if st.session_state.cached_weak_areas:
                    print("📋 Слабые места загружены в кэш при старте")
            except:
                pass
    
    error_analysis_toggle = st.checkbox(
        "Enable adaptive learning",
        value=st.session_state.error_analysis_enabled,
        help="AI will analyze your mistakes and focus on weak areas"
    )
    
    if error_analysis_toggle != st.session_state.error_analysis_enabled:
        st.session_state.error_analysis_enabled = error_analysis_toggle
        if error_analysis_toggle and st.session_state.api_key_input:
            st.session_state.error_memory_app = create_error_memory_app(st.session_state.api_key_input)
        st.rerun()
    
    if st.session_state.error_analysis_enabled and st.session_state.error_memory_app:
        if st.button("📊 View My Weak Areas"):
            with st.spinner("Analyzing your learning patterns..."):
                weak_areas = get_weak_areas_context(st.session_state.error_memory_app)
                if weak_areas:
                    st.info("**Your focus areas:**")
                    st.write(weak_areas)
                else:
                    st.success("No patterns yet. Keep practicing!")
        
        if st.button("🗑️ Reset Learning History"):
            st.session_state.error_memory_app.reset()
            st.success("Learning history cleared!")
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📄 Upload PDF (Optional)")
    st.markdown("Upload an article or document to discuss during conversation")
    
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
    
    if pdf_file is not None:
        if st.session_state.pdf_name != pdf_file.name:
            with st.spinner("📖 Processing PDF with RAG..."):
                if st.session_state.rag_app is not None:
                    try:
                        st.session_state.rag_app.reset()
                        st.info("🗑️ Cleared previous PDF from database")
                    except Exception as e:
                        logger.error(f"Error resetting RAG app: {str(e)}")
                        st.session_state.rag_app = None
                
                if st.session_state.rag_app is None:
                    st.session_state.rag_app = create_rag_app(st.session_state.api_key_input)
                
                if st.session_state.rag_app:
                    success = add_pdf_to_rag(st.session_state.rag_app, pdf_file)
                    if success:
                        st.session_state.pdf_name = pdf_file.name
                        st.success(f"✅ Loaded: {pdf_file.name} (RAG ready)")
                    else:
                        st.error("Failed to process PDF")
                else:
                    st.error("Failed to initialize RAG system")
    
    if st.session_state.pdf_name:
        if st.button("🗑️ Remove PDF"):
            st.session_state.rag_app = None
            st.session_state.pdf_name = None
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🔄 Reset Conversation"):
        st.session_state.messages = []
        st.session_state.last_audio_id = None
        st.session_state.played_audio_ids = set()
        st.session_state.show_feedback_panel = False
        st.rerun()

st.title("Speech-to-Speech English Practice")

col_title, col_button = st.columns([3, 1])
with col_title:
    subtitle = "Have a **natural spoken conversation** in English!"
    if st.session_state.error_analysis_enabled:
        subtitle += " 🧠 *Adaptive learning: ON*"
    if st.session_state.pdf_name:
        subtitle += f" | 📄 *{st.session_state.pdf_name}*"
    st.markdown(subtitle)
with col_button:
    if st.button("📝 Transcript & Feedback" if not st.session_state.show_feedback_panel else "✖ Close Panel", 
                 use_container_width=True,
                 type="secondary" if not st.session_state.show_feedback_panel else "primary"):
        st.session_state.show_feedback_panel = not st.session_state.show_feedback_panel
        st.rerun()

if st.session_state.show_feedback_panel:
    col1, col2 = st.columns([1, 1])
else:
    col1 = st.container()
    col2 = None

with col1:
    st.markdown("### 🎙️ Voice Conversation")
    
    if len(st.session_state.messages) == 0:
        welcome_msg = "👋 **Hi there!** I'm your English conversation partner. Press the button below to start recording and say something!"
        if st.session_state.error_analysis_enabled:
            welcome_msg += "\n\n🧠 **Adaptive Learning is ON:** I'll analyze your speech patterns and automatically focus on areas where you need more practice!"
        if st.session_state.pdf_name:
            welcome_msg += f"\n\n📄 **We'll be discussing:** {st.session_state.pdf_name}"
        st.info(welcome_msg)
    
    st.markdown("---")
    recorded_audio = st.audio_input("🎤 Click to record your message")
    
    user_input = None
    user_transcription = None
    
    if recorded_audio and st.session_state.api_key_input:
        audio_id = hash(recorded_audio.getvalue())
        
        if audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id
            
            if not st.session_state.processing:
                st.session_state.processing = True
                
                st.session_state.total_start_time = time.time()
                print("="*60)
                print("🚀 НАЧАЛО ПОЛНОГО ЦИКЛА ОБРАБОТКИ")
                print("="*60)
                
                with st.spinner("🎧 Listening..."):
                    whisper = load_whisper_model()
                    if whisper:
                        user_transcription = transcribe_audio(recorded_audio, whisper)
                        if user_transcription:
                            user_input = user_transcription
                            st.success("✓ Got it!")
    
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant" and "audio_path" in message and os.path.exists(message["audio_path"]):
            audio_id = f"audio_{i}"
            should_autoplay = audio_id not in st.session_state.played_audio_ids
            
            if should_autoplay:
                st.session_state.played_audio_ids.add(audio_id)
                audio_html = get_audio_html(message["audio_path"], autoplay=True, show_controls=False)
                if audio_html:
                    st.markdown(audio_html, unsafe_allow_html=True)

if st.session_state.show_feedback_panel and col2:
    with col2:
        st.markdown("### 📝 Transcript & Feedback")
        if len(st.session_state.messages) == 0:
            st.info("No conversation transcript yet. Start talking to get started!")
        
        container_placeholder = st.container(height=400)
        
        with container_placeholder:
            if len(st.session_state.messages) > 0:
                for i, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        st.markdown("**You:**")
                        st.markdown(f"*{message['content']}*")
                        
                        if i + 1 < len(st.session_state.messages):
                            next_message = st.session_state.messages[i + 1]
                            if next_message["role"] == "assistant":
                                conversation, feedback = parse_agent_response(next_message["content"])
                                
                                if feedback and feedback.strip() and feedback != "Great job!":
                                    st.markdown("💡 **Feedback:**")
                                    st.markdown(f"<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 5px;'>{feedback}</div>", unsafe_allow_html=True)
                                else:
                                    st.success("✨ Great job!")
                        
                        st.markdown("---")
                    else:
                        st.markdown("**AI Partner:**")
                        conversation, feedback = parse_agent_response(message["content"])
                        st.markdown(f"*{conversation}*")
                        st.markdown("---")

if user_input and st.session_state.processing:
    if not st.session_state.api_key_input:
        st.warning("⚠️ Please enter your DeepSeek API key!")
        st.session_state.processing = False
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        
        
        with st.spinner("💭 Thinking..."):
            pdf_context = ""
            if st.session_state.rag_app and st.session_state.pdf_name:
                rag_response = query_rag(st.session_state.rag_app, user_input)
                if rag_response:
                    pdf_context = f"\n\n📄 Relevant information from the document:\n{rag_response}\n"
            
            if _background_cache["weak_areas"] is not None:
                st.session_state.cached_weak_areas = _background_cache["weak_areas"]
                print(f"📋 Слабые места обновлены из фонового кэша (0 сек!)")
            
            weak_areas_context = st.session_state.cached_weak_areas
            
            if weak_areas_context:
                print(f"📋 Используем КЭШИРОВАННЫЕ слабые места")
            else:
                print(f"📋 Слабые места: кэш пуст (обновится после ответа)")
            
            agent = initialize_conversation_agent(
                st.session_state.api_key_input,
                has_pdf=bool(st.session_state.pdf_name),
                weak_areas=weak_areas_context
            )
            
            if agent:
                conversation_context = "Previous conversation:\n"
                for msg in st.session_state.messages[-6:]:
                    role = "Student" if msg["role"] == "user" else "Teacher"
                    conversation_context += f"{role}: {msg['content']}\n"
                
                conversation_context += f"\nStudent just said: {user_input}"
                
                if pdf_context:
                    conversation_context += pdf_context
                
                conversation_context += "\n\nRespond naturally."
                
                
                tts = load_tts_model()
                
                agent_start_time = time.time()
                print("⏳ Генерация ответа AI агента...")
                
                
                tts_futures = {}
                processed_sentences = set()
                executor = ThreadPoolExecutor(max_workers=4)
                streaming_worked = False
                
                try:
                    response_stream = agent.run(conversation_context, stream=True)
                    
                    full_response = ""
                    feedback_started = False
                    
                    for chunk in response_stream:
                        if hasattr(chunk, 'content') and chunk.content:
                            full_response += chunk.content
                            
                            if "FEEDBACK:" in full_response.upper() or "FEEDBACK :" in full_response.upper():
                                if not feedback_started:
                                    feedback_started = True
                                    print(f"📝 [STREAMING] FEEDBACK секция обнаружена - TTS остановлен")
                            
                            if tts and not feedback_started:
                                conversation_part = full_response
                                if "CONVERSATION:" in conversation_part.upper():
                                    conv_start = conversation_part.upper().find("CONVERSATION:")
                                    conversation_part = conversation_part[conv_start + 13:].strip()
                                
                                current_sentences = split_into_sentences(conversation_part)
                                
                                for idx, sentence in enumerate(current_sentences[:-1]):
                                    if idx not in processed_sentences and len(sentence) > 10:
                                        processed_sentences.add(idx)
                                        print(f"🔀 [STREAMING] Предложение {idx + 1} получено → TTS запущен")
                                        
                                        tts_futures[idx] = executor.submit(
                                            text_to_speech_raw, sentence, tts
                                        )
                    
                    conversation_text, feedback_text = parse_agent_response(full_response)
                    response_content = full_response
                    
                    if tts:
                        final_sentences = split_into_sentences(conversation_text)
                        for idx, sentence in enumerate(final_sentences):
                            if idx not in processed_sentences and len(sentence) > 10:
                                processed_sentences.add(idx)
                                print(f"🔀 [STREAMING] Предложение {idx + 1} (финальное) → TTS запущен")
                                tts_futures[idx] = executor.submit(
                                    text_to_speech_raw, sentence, tts
                                )
                    
                    agent_elapsed = time.time() - agent_start_time
                    print(f"✅ Ответ AI агента получен за {agent_elapsed:.2f} секунд (стриминг)")
                    print(f"   └── {len(tts_futures)} TTS потоков запущено параллельно")
                    
                    streaming_worked = len(tts_futures) > 0
                    
                except (TypeError, AttributeError) as e:
                    print(f"   └── Стриминг не поддерживается, используем обычный режим")
                    response = agent.run(conversation_context)
                    agent_elapsed = time.time() - agent_start_time
                    print(f"✅ Ответ AI агента получен за {agent_elapsed:.2f} секунд")
                    
                    conversation_text, feedback_text = parse_agent_response(response.content)
                    response_content = response.content
                    streaming_worked = False
                
                with st.spinner("🗣️ Speaking..."):
                    if tts:
                        if streaming_worked and tts_futures:
                            print(f"⏳ Ожидание готовности {len(tts_futures)} TTS потоков...")
                            
                            audio_results = {}
                            
                            for idx in sorted(tts_futures.keys()):
                                future = tts_futures[idx]
                                try:
                                    audio = future.result(timeout=30)
                                    if audio is not None:
                                        audio_results[idx] = audio
                                        print(f"   ✓ Предложение {idx + 1} готово")
                                except Exception as e:
                                    logger.error(f"TTS future {idx} failed: {e}")
                            
                            if audio_results:
                                ordered_audio = [audio_results[i] for i in sorted(audio_results.keys())]
                                full_audio = np.concatenate(ordered_audio) if len(ordered_audio) > 1 else ordered_audio[0]
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                                    sf.write(tmp_file.name, full_audio, 24000)
                                    audio_path = tmp_file.name
                                print(f"✅ Речь готова ({len(audio_results)} предложений, параллельная генерация)")
                            else:
                                audio_path = None
                        else:
                            audio_path = text_to_speech_parallel(conversation_text, tts, max_workers=3)
                        
                        executor.shutdown(wait=False)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_content,
                            "audio_path": audio_path
                        })
                    else:
                        st.error("Failed to generate speech")
            else:
                st.error("Failed to initialize AI agent. Please check your API key.")
        
        st.session_state.processing = False
        
        if st.session_state.total_start_time:
            total_elapsed = time.time() - st.session_state.total_start_time
            print("="*60)
            print(f"✅ ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН за {total_elapsed:.2f} секунд")
            print("="*60)
            st.session_state.total_start_time = None
        
        if st.session_state.error_analysis_enabled and st.session_state.error_memory_app:
            _user_input = user_input
            _api_key = st.session_state.api_key_input
            _error_memory_app = st.session_state.error_memory_app
            
            def post_response_background_tasks(text, api_key, memory_app):
                """Run error analysis and update weak areas AFTER response"""
                global _background_cache
                try:
                    print("🔄 [POST-RESPONSE] Анализ ошибок пользователя...")
                    analyze_user_errors(text, api_key, memory_app)
                    print("✅ [POST-RESPONSE] Анализ ошибок завершен")
                    
                    print("🔄 [POST-RESPONSE] Обновление кэша слабых мест...")
                    new_weak_areas = get_weak_areas_context(memory_app)
                    
                    _background_cache["weak_areas"] = new_weak_areas
                    _background_cache["last_updated"] = time.time()
                    print("✅ [POST-RESPONSE] Глобальный кэш слабых мест обновлен")
                    
                except Exception as e:
                    logger.error(f"Error in post-response tasks: {str(e)}")
            
            background_thread = Thread(
                target=post_response_background_tasks,
                args=(_user_input, _api_key, _error_memory_app),
                daemon=True
            )
            background_thread.start()
            print("🔄 [POST-RESPONSE] Фоновые задачи запущены (пока пользователь слушает)")
        
        st.rerun()

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Practice makes perfect! Keep chatting to improve your English.</p>
    <p style='font-size: 0.9em;'>Powered by DeepSeek AI, Whisper & Kokoro TTS</p>
</div>
""", unsafe_allow_html=True)
