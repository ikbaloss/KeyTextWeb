import os
import re
import io
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import collections
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gensim.models import Word2Vec
import streamlit as st

# ==========================================
# 1. Page Configuration & State Reset
# ==========================================
st.set_page_config(page_title="KeyText Version 0.22", layout="wide")
st.title("🔑 KeyText Version 0.22")

# Replicating comprehensive desktop instance states inside memory buffers
if "main_data" not in st.session_state:
    st.session_state.main_data = pd.DataFrame()
if "unigrams" not in st.session_state:
    st.session_state.unigrams = pd.DataFrame()
if "bigrams" not in st.session_state:
    st.session_state.bigrams = pd.DataFrame()
if "stop_words" not in st.session_state:
    st.session_state.stop_words = []
if "word_freq_dict" not in st.session_state:
    st.session_state.word_freq_dict = defaultdict(list)
if "prev_word_freq_dict" not in st.session_state:
    st.session_state.prev_word_freq_dict = defaultdict(list)
if "wv_model" not in st.session_state:
    st.session_state.wv_model = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Persistent structural storage configurations for tab environments
for key in ["kw1", "kw2", "kw3", "lbl1", "lbl2", "lbl3", "search_term", "kwic_term"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ==========================================
# 2. Algorithmic Helper Functions
# ==========================================
def keep_alphanumeric(input_string):
    alphanumeric_chars = [char if char.isalnum() or char=='_' or char=='-' else ' ' for char in input_string]
    result_string = ''.join(alphanumeric_chars)
    if re.search(r'[a-zA-Z0-9]', result_string):
        return result_string
    return ""

def wholeword(text, keywords):
    pattern = r"\b(?:\w)+"
    words = re.findall(pattern, text.lower())
    return any(keyword in words for keyword in keywords)

def fill_unigrams(df):
    if 'SelectedColumn' in df.columns:
        text = " ".join(df['SelectedColumn'].dropna().astype(str).tolist())
        tokens = re.findall(r'\b\w+(?:[-_]\w+)*\b', text.lower())
        tokens = [w for w in tokens if w not in st.session_state.stop_words]
        unigram_freq = Counter(tokens)
        unigrams_df = pd.DataFrame(unigram_freq.items(), columns=['Unigram', 'Frequency'])
        return unigrams_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return pd.DataFrame()

def fill_bigrams(df):
    if 'SelectedColumn' in df.columns:
        text = " ".join(df['SelectedColumn'].dropna().astype(str).tolist())
        tokens = re.findall(r'\b\w+(?:[-_]\w+)*\b', text.lower())
        tokens = [w for w in tokens if w not in st.session_state.stop_words]
        bigrams_list = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        bigram_freq = Counter(bigrams_list)
        bigrams_df = pd.DataFrame(bigram_freq.items(), columns=['Bigram', 'Frequency'])
        return bigrams_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return pd.DataFrame()

# ==========================================
# 3. Sidebar Actions (File Pipeline Loader)
# ==========================================
with st.sidebar:
    st.header("📁 File System Actions")
    uploaded_files = st.file_uploader("Open CSV or TXT Files", accept_multiple_files=True, type=["csv", "txt"])
    
    txt_delimiter = st.text_input("Paragraph Delimiter (for TXT only)", value="\\n")
    txt_delimiter = txt_delimiter.replace('\\n', '\n')

    if st.button("🚀 Process & Load Files", use_container_width=True):
        if not uploaded_files:
            st.error("No data engine files found. Upload vectors.")
        else:
            list_of_files = []
            file_types = [f.name.split('.')[-1] for f in uploaded_files]
            
            if len(set(file_types)) > 1:
                st.error("All batch processing entities must utilize identical extensions.")
            else:
                is_csv = file_types[0] == 'csv'
                
                for idx, f in enumerate(uploaded_files):
                    fileName = os.path.splitext(f.name)[0]
                    if is_csv:
                        # Fixed binary upload text wrapping context using TextIOWrapper
                        text_stream = io.TextIOWrapper(f, encoding='utf-8', errors='backslashreplace')
                        dfcsv = pd.read_csv(text_stream)
                        if len(uploaded_files) > 1:
                            dfcsv.insert(loc=1, column='Data', value=[fileName]*len(dfcsv))
                        list_of_files.append(dfcsv)
                    else:
                        raw_bytes = f.read()
                        raw_text = raw_bytes.decode('utf-8', errors='backslashreplace')
                        paragraphs = raw_text.split(txt_delimiter)
                        pars = [" ".join(p.split()) for p in paragraphs if p.strip()]
                        
                        if len(uploaded_files) > 1:
                            dftext = pd.DataFrame({'Text': pars, 'Data': [fileName]*len(pars)})
                        else:
                            dftext = pd.DataFrame({'Text': pars})
                        
                        n_rows = len(dftext)
                        n_dates = min(100, n_rows)
                        if n_dates > 0:
                            group_size = n_rows // n_dates
                            date_assignments = []
                            for i in range(n_dates):
                                current_size = group_size + (1 if i < (n_rows % n_dates) else 0)
                                date_assignments.extend([i] * current_size)
                            all_dates = [datetime.now().date() - timedelta(days=(n_dates - 1 - k)) for k in range(n_dates)]
                            dftext['Date'] = [all_dates[k] for k in date_assignments]
                        else:
                            dftext['Date'] = datetime.now().date()
                            
                        list_of_files.append(dftext)
                
                df_combined = pd.concat(list_of_files, ignore_index=True)
                df_combined.drop_duplicates(inplace=True)
                st.session_state.main_data = df_combined
                st.success(f"Successfully assembled {len(df_combined)} unique workspace vectors.")

# ==========================================
# 4. Tab Environment Matrix Transformation
# ==========================================
tabs = st.tabs([
    "📊 Raw Data Workspace", 
    "🔍 Word2Vec Vector Space", 
    "📖 KWIC Explorer", 
    "📈 Trend Comparisons", 
    "🔠 N-Gram Matrix", 
    "🕸️ Co-occurrence Modeler"
])

# ------------------------------------------
# TAB 1: RAW DATA WORKSPACE
# ------------------------------------------
with tabs[0]:
    st.header("Raw Target Matrices Configuration")
    if st.session_state.main_data.empty:
        st.info("Awaiting structural document vectors from file source panel loading arrays.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            lang_choice = st.selectbox("Language Model Selection", options=["Indonesia", "English"])
        with col2:
            columns_list = list(st.session_state.main_data.columns)
            date_col_choice = st.selectbox("Target Date Coordinates", options=["Select"] + columns_list, index=columns_list.index("Date") + 1 if "Date" in columns_list else 0)
        with col3:
            day_first_chk = st.checkbox("Day First Datetime Flag", value=False)
        with col4:
            text_col_choice = st.selectbox("Target Linguistic Column", options=["Select"] + columns_list, index=columns_list.index("Text") + 1 if "Text" in columns_list else 0)
            
        if st.button("⚡ Execute Computational Tokenization & Training", type="primary", use_container_width=True):
            if text_col_choice == "Select":
                st.warning("Computational pipeline requires explicit linguistic target matrix selection column alignment.")
            else:
                df = st.session_state.main_data.copy()
                df['SelectedColumn'] = df[text_col_choice].astype(str).str.lower()
                
                if date_col_choice != 'Select':
                    df = df.rename(columns={date_col_choice: 'Date'})
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=day_first_chk, errors='coerce').dt.date
                
                df['SelectedColumn'] = df['SelectedColumn'].fillna('')
                comments = [s for s in df['SelectedColumn'].to_list() if isinstance(s, str) and s.strip() != '']
                cleaned_comments = [keep_alphanumeric(s) for s in comments]
                token_comments = [s.split() for s in cleaned_comments]
                
                # Fetching Stopwords definitions
                stopword_file = "stopwords-id.txt" if lang_choice == "Indonesia" else "stopwords-en.txt"
                if os.path.exists(stopword_file):
                    with open(stopword_file, "r") as tf:
                        st.session_state.stop_words = tf.read().split()
                else:
                    st.session_state.stop_words = []
                
                with st.spinner("Compiling Word2Vec Vector Space Nodes..."):
                    wv_model = Word2Vec(sentences=token_comments, min_count=5, vector_size=100, window=5, sg=1)
                
                word_freq = defaultdict(list)
                prev_word_freq = defaultdict(list)
                for text in comments:
                    chunks = re.split(r'[^\w\s\-_]', text.lower())
                    for chunk in chunks:
                        words = re.findall(r'\b\w+(?:[-_]\w+)*\b', chunk)
                        for i, word in enumerate(words):
                            if i + 1 < len(words):
                                word_freq[word].append(words[i + 1])
                            if i > 0:
                                prev_word_freq[word].append(words[i - 1])
                                
                st.session_state.word_freq_dict = word_freq
                st.session_state.prev_word_freq_dict = prev_word_freq
                st.session_state.wv_model = wv_model
                st.session_state.unigrams = fill_unigrams(df)
                st.session_state.bigrams = fill_bigrams(df)
                st.session_state.main_data = df
                st.session_state.processing_done = True
                st.balloons()
                
        st.dataframe(st.session_state.main_data, use_container_width=True)

# ------------------------------------------
# TAB 2: WORD2VEC VECTOR SPACE EXPLORER
# ------------------------------------------
with tabs[1]:
    st.header("🔍 Neural Semantic Space Search Engine")
    if not st.session_state.processing_done:
        st.info("Initialize raw textual arrays inside Tab 1 matrix processors first.")
    else:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            st.session_state.search_term = st.text_input("Enter Target Semantic Token Vector", value=st.session_state.search_term)
        with sc2:
            top_n = st.spinbox = st.number_input("Top Match Bounds Limit", min_value=5, max_value=100, value=20)
            
        if st.session_state.search_term:
            cleaned_search = st.session_state.search_term.lower().strip()
            if st.session_state.wv_model and cleaned_search in st.session_state.wv_model.wv:
                sim_results = st.session_state.wv_model.wv.most_similar(cleaned_search, topn=top_n)
                sim_df = pd.DataFrame(sim_results, columns=['Associated Node Token', 'Cosine Metric Coordinate Score'])
                
                st.subheader(f"Vector Space Neighbors mapping: '{cleaned_search}'")
                st.dataframe(sim_df, use_container_width=True)
            else:
                st.error(f"Token structural definition node '{cleaned_search}' not established within model vector space vocabulary matrices.")

# ------------------------------------------
# TAB 3: KEY WORD IN CONTEXT (KWIC) EXPLORER
# ------------------------------------------
with tabs[2]:
    st.header("📖 Sliced Window Context Extraction (KWIC)")
    if not st.session_state.processing_done:
        st.info("Initialize raw textual arrays inside Tab 1 matrix processors first.")
    else:
        kc1, kc2 = st.columns([3, 1])
        with kc1:
            st.session_state.kwic_term = st.text_input("Contextual Window Key Target Expression String", value=st.session_state.kwic_term)
        with kc2:
            window_size = st.number_input("Window Byte Length Context Truncation Radius", min_value=2, max_value=15, value=5)
            
        if st.session_state.kwic_term:
            target_kwic = st.session_state.kwic_term.lower().strip()
            kwic_records = []
            
            for sentence in st.session_state.main_data['SelectedColumn'].dropna().astype(str):
                tokens = re.findall(r'\b\w+(?:[-_]\w+)*\b', sentence)
                for index, token in enumerate(tokens):
                    if token == target_kwic:
                        left_bound = max(0, index - window_size)
                        right_bound = min(len(tokens), index + window_size + 1)
                        
                        kwic_records.append({
                            "Left Node Frame Context": " ".join(tokens[left_bound:index]),
                            "TARGET KEYWORD NODE": tokens[index],
                            "Right Node Frame Context": " ".join(tokens[index+1:right_bound])
                        })
                        
            if kwic_records:
                kwic_df = pd.DataFrame(kwic_records)
                st.dataframe(kwic_df, use_container_width=True)
            else:
                st.warning("No localized matching contextual windows surfaced along active evaluation text matrices.")

# ------------------------------------------
# TAB 4: TREND COMPARISONS
# ------------------------------------------
with tabs[3]:
    st.header("📈 Longitudinal Vector Cohort Comparison Filters")
    if not st.session_state.processing_done:
        st.info("Initialize raw textual arrays inside Tab 1 matrix processors first.")
    else:
        r1c1, r1c2 = st.columns([3, 2])
        with r1c1: st.session_state.kw1 = st.text_input("Keywords Sequence Array Cluster Group 1 (pipe | separated)", value=st.session_state.kw1)
        with r1c2: st.session_state.lbl1 = st.text_input("Display Label Mapping Node Alias 1", value=st.session_state.lbl1)
            
        r2c1, r2c2 = st.columns([3, 2])
        with r2c1: st.session_state.kw2 = st.text_input("Keywords Sequence Array Cluster Group 2 (pipe | separated)", value=st.session_state.kw2)
        with r2c2: st.session_state.lbl2 = st.text_input("Display Label Mapping Node Alias 2", value=st.session_state.lbl2)

        r3c1, r3c2 = st.columns([3, 2])
        with r3c1: st.session_state.kw3 = st.text_input("Keywords Sequence Array Cluster Group 3 (pipe | separated)", value=st.session_state.kw3)
        with r3c2: st.session_state.lbl3 = st.text_input("Display Label Mapping Node Alias 3", value=st.session_state.lbl3)

        if st.button("📊 Render Longitudinal Cohort Vector Line Visualizations", type="primary", use_container_width=True):
            df = st.session_state.main_data.copy()
            keywords1 = [w.strip() for w in st.session_state.kw1.split('|') if w.strip() != '']
            keywords2 = [w.strip() for w in st.session_state.kw2.split('|') if w.strip() != '']
            keywords3 = [w.strip() for w in st.session_state.kw3.split('|') if w.strip() != '']
            
            if not (keywords1 or keywords2 or keywords3):
                st.warning("Longitudinal analysis sequence blocks missing keyword targets.")
            else:
                all_labels = []
                for idx, (kws, lbl_val, fallback) in enumerate([(keywords1, st.session_state.lbl1, 'Group 1'), 
                                                             (keywords2, st.session_state.lbl2, 'Group 2'), 
                                                             (keywords3, st.session_state.lbl3, 'Group 3')]):
                    if kws:
                        target_label = lbl_val.strip() if lbl_val.strip() != '' else fallback
                        df[target_label] = df['SelectedColumn'].apply(lambda x: 1 if wholeword(x, kws) else 0)
                        all_labels.append(target_label)
                
                df_summed = df.groupby('Date')[all_labels].sum().reset_index()
                start_date, end_date = df_summed['Date'].min(), df_summed['Date'].max()
                
                fig, ax = plt.subplots(figsize=(12, 5))
                for label in all_labels:
                    ax.plot(df_summed['Date'], df_summed[label], marker='o', label=label)
                    
                ax.set_ylabel('Absolute Cluster Hit Frequencies')
                ax.set_xlabel('Timeline Interval Index Points')
                ax.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.dataframe(df_summed, use_container_width=True)

# ------------------------------------------
# TAB 5: N-GRAM MATRIX EXTRAPOLATOR
# ------------------------------------------
with tabs[4]:
    st.header("🔠 Structural Segment N-Gram Vector Matrices")
    if not st.session_state.processing_done:
        st.info("Initialize raw textual arrays inside Tab 1 matrix processors first.")
    else:
        nc1, nc2 = st.columns(2)
        with nc1:
            st.subheader("Top Extracted Unigram Distribution Core Frame")
            st.dataframe(st.session_state.unigrams, use_container_width=True)
        with nc2:
            st.subheader("Top Extracted Bigram Distribution Core Frame")
            st.dataframe(st.session_state.bigrams, use_container_width=True)

# ------------------------------------------
# TAB 6: NETWORK CO-OCCURRENCE MODELER
# ------------------------------------------
with tabs[5]:
    st.header("🕸️ Relational Structural Token Co-occurrence Matrices")
    if not st.session_state.processing_done:
        st.info("Initialize raw textual arrays inside Tab 1 matrix processors first.")
    else:
        st.write("Construct relational adjacency edges maps by specifying target central focus index bounds:")
        co_target = st.text_input("Focal Node Element Token Anchor", value=st.session_state.search_term if st.session_state.search_term else "text")
        
        if co_target:
            t_clean = co_target.lower().strip()
            next_nodes = st.session_state.word_freq_dict.get(t_clean, [])
            prev_nodes = st.session_state.prev_word_freq_dict.get(t_clean, [])
            
            all_neighbors = next_nodes + prev_nodes
            if all_neighbors:
                neighbor_counts = Counter(all_neighbors)
                edge_df = pd.DataFrame(neighbor_counts.items(), columns=['Adjacency Node Link', 'Co-occurrence Frequency Weight'])
                edge_df = edge_df.sort_values(by='Co-occurrence Frequency Weight', ascending=False).reset_index(drop=True)
                
                st.subheader(f"Adjacency Association Vectors mapped for focus structural root: '{t_clean}'")
                st.dataframe(edge_df, use_container_width=True)
                
                # Dynamic GML/CSV structural export button replacing desktop memory downloads
                csv_buffer = edge_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"💾 Download Adjacency Graph Edge Arrays List for '{t_clean}' (CSV)",
                    data=csv_buffer,
                    file_name=f"edges_{t_clean}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No direct graph neighborhood pathways map to: '{t_clean}'")

# ==========================================
# 5. Application Structural Footer
# ==========================================
st.markdown("---")
st.caption("Copyright ©2026 Ikbal Maulana • Unified Content Analyzer Pipeline System Engine")