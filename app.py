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
# 1. Page Configuration & State Setup
# ==========================================
st.set_page_config(page_title="KeyText Version 0.22", layout="wide")
st.title("🔑 KeyText Version 0.22")

# Replicating original persistent memory storage instances
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
if "df_graph_to_save" not in st.session_state:
    st.session_state.df_graph_to_save = pd.DataFrame()

# Persistent state buffers for keyword analysis tabs
for key in ["kw1", "kw2", "kw3", "lbl1", "lbl2", "lbl3", "search_term", "kwic_term"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ==========================================
# 2. Algorithmic Helper Methods
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
# 3. Sidebar Actions (File Pipeline System)
# ==========================================
with st.sidebar:
    st.header("📁 File Upload System")
    uploaded_files = st.file_uploader("Open CSV or TXT Files", accept_multiple_files=True, type=["csv", "txt"])
    
    txt_delimiter = st.text_input("Paragraph Delimiter (for TXT files)", value="\\n")
    txt_delimiter = txt_delimiter.replace('\\n', '\n')

    if st.button("🚀 Load Uploaded Files", use_container_width=True):
        if not uploaded_files:
            st.error("Please provide at least one source document vector file.")
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
                st.session_state.processing_done = False # Reset flag for new file configurations
                st.success(f"Assembled {len(df_combined)} records. Configure target columns below.")

# ==========================================
# 4. Global Dropdown Configurations Matrix
# ==========================================
columns_list = ["Select"] + list(st.session_state.main_data.columns) if not st.session_state.main_data.empty else ["Select"]

st.subheader("⚙️ Meta Configuration Setup Control Panel")
col1, col2, col3, col4 = st.columns(4)

with col1:
    lang_choice = st.selectbox("Language Selection", options=["Indonesia", "English"])
with col2:
    date_col_choice = st.selectbox("Date Identifier Column", options=columns_list, index=columns_list.index("Date") if "Date" in columns_list else 0)
with col3:
    day_first_chk = st.checkbox("Day First Format Flag", value=False)
with col4:
    text_col_choice = st.selectbox("Text Analysis Column Target", options=columns_list, index=columns_list.index("Text") if "Text" in columns_list else 0)

if st.button("⚡ Select Text & Compile Word Embeddings (Word2Vec)", type="primary", use_container_width=True):
    if text_col_choice == "Select":
        st.error("Linguistic analytics pipeline requires an assigned Text Analysis Target Column.")
    else:
        with st.spinner("Processing text and compiling Word2Vec models..."):
            df = st.session_state.main_data.copy()
            df['SelectedColumn'] = df[text_col_choice].astype(str).str.lower()
            
            if date_col_choice != 'Select':
                df = df.rename(columns={date_col_choice: 'Date'})
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=day_first_chk, errors='coerce').dt.date
            
            df['SelectedColumn'] = df['SelectedColumn'].fillna('')
            comments = [s for s in df['SelectedColumn'].to_list() if isinstance(s, str) and s.strip() != '']
            cleaned_comments = [keep_alphanumeric(s) for s in comments]
            token_comments = [s.split() for s in cleaned_comments]
            
            # Stopwords Handler Matrix
            stopword_file = "stopwords-id.txt" if lang_choice == "Indonesia" else "stopwords-en.txt"
            if os.path.exists(stopword_file):
                with open(stopword_file, "r") as tf:
                    st.session_state.stop_words = tf.read().split()
            else:
                st.session_state.stop_words = []
            
            # Original desktop parameters: min_count=20, vector_size=200, window=3, sg=1
            wv_model = Word2Vec(sentences=token_comments, min_count=20, vector_size=200, window=3, sg=1)
            
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
            st.success("Analysis pipeline processed. Active workspaces are ready.")

# ==========================================
# 5. Tab Environment Engine Matrix
# ==========================================
tabs = st.tabs([
    "📊 Raw Data Dataset", 
    "🔍 Word2Vec Embeddings", 
    "📖 KWIC Explorer", 
    "📈 Trend Comparisons", 
    "🔠 N-Gram Distributions", 
    "🕸️ Co-occurrence Modeler"
])

# ------------------------------------------
# TAB 1: RAW DATA VIEW
# ------------------------------------------
with tabs[0]:
    if st.session_state.main_data.empty:
        st.info("Awaiting structural document vectors from file source loading panel.")
    else:
        st.dataframe(st.session_state.main_data, use_container_width=True)

# ------------------------------------------
# TAB 2: WORD2VEC SIMILARITY MATRIX
# ------------------------------------------
with tabs[1]:
    st.header("🔍 Neural Semantic Space Engine")
    if not st.session_state.processing_done:
        st.info("Execute 'Select Text' matrix compilation above to unlock this space.")
    else:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            st.session_state.search_term = st.text_input("Search Keyword Vector Embeddings", value=st.session_state.search_term)
        with sc2:
            top_n = st.number_input("Top Matches Bound Limit", min_value=5, max_value=100, value=20)
            
        if st.session_state.search_term:
            cleaned_search = st.session_state.search_term.lower().strip()
            if st.session_state.wv_model and cleaned_search in st.session_state.wv_model.wv:
                sim_results = st.session_state.wv_model.wv.most_similar(cleaned_search, topn=top_n)
                sim_df = pd.DataFrame(sim_results, columns=['Associated Token Node', 'Cosine Metric Score Value'])
                st.subheader(f"Vector Space Neighbor Results: '{cleaned_search}'")
                st.dataframe(sim_df, use_container_width=True)
            else:
                st.error(f"Token node '{cleaned_search}' not established within model vocabulary matrices.")

# ------------------------------------------
# TAB 3: KWIC CONTEXTUAL SLICER
# ------------------------------------------
with tabs[2]:
    st.header("📖 Key Word In Context Window (KWIC)")
    if not st.session_state.processing_done:
        st.info("Execute 'Select Text' matrix compilation above to unlock this space.")
    else:
        kc1, kc2 = st.columns([3, 1])
        with kc1:
            st.session_state.kwic_term = st.text_input("Contextual Analysis Target String Expression", value=st.session_state.kwic_term)
        with kc2:
            window_size = st.number_input("Contextual Token Truncation Radius Window", min_value=2, max_value=15, value=5)
            
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
                            "Left Context Frame": " ".join(tokens[left_bound:index]),
                            "TARGET KEYWORD NODE": tokens[index],
                            "Right Context Frame": " ".join(tokens[index+1:right_bound])
                        })
                        
            if kwic_records:
                st.dataframe(pd.DataFrame(kwic_records), use_container_width=True)
            else:
                st.warning("No context intervals generated for selection coordinates.")

# ------------------------------------------
# TAB 4: CATEGORY TREND COMPARISONS
# ------------------------------------------
with tabs[3]:
    st.header("📈 Longitudinal Trend Line Plots")
    if not st.session_state.processing_done:
        st.info("Execute 'Select Text' matrix compilation above to unlock this space.")
    else:
        r1c1, r1c2 = st.columns([3, 2])
        with r1c1: st.session_state.kw1 = st.text_input("Keywords Group 1 (separated by pipe |)", value=st.session_state.kw1)
        with r1c2: st.session_state.lbl1 = st.text_input("Display Label Assignment 1", value=st.session_state.lbl1)
            
        r2c1, r2c2 = st.columns([3, 2])
        with r2c1: st.session_state.kw2 = st.text_input("Keywords Group 2 (separated by pipe |)", value=st.session_state.kw2)
        with r2c2: st.session_state.lbl2 = st.text_input("Display Label Assignment 2", value=st.session_state.lbl2)

        r3c1, r3c2 = st.columns([3, 2])
        with r3c1: st.session_state.kw3 = st.text_input("Keywords Group 3 (separated by pipe |)", value=st.session_state.kw3)
        with r3c2: st.session_state.lbl3 = st.text_input("Display Label Assignment 3", value=st.session_state.lbl3)

        graph_type = st.radio("Visualization Filter Format Pattern", options=["Side by Side Plot", "Filtered Accumulations"])

        if st.button("📊 Render Trend Visualization Plot Graph", type="primary", use_container_width=True):
            df = st.session_state.main_data.copy()
            keywords1 = [w.strip() for w in st.session_state.kw1.split('|') if w.strip() != '']
            keywords2 = [w.strip() for w in st.session_state.kw2.split('|') if w.strip() != '']
            keywords3 = [w.strip() for w in st.session_state.kw3.split('|') if w.strip() != '']
            
            if not (keywords1 or keywords2 or keywords3):
                st.warning("Trend configuration analysis blocks missing expression keyword targets.")
            else:
                all_labels = []
                for idx, (kws, lbl_val, fallback) in enumerate([(keywords1, st.session_state.lbl1, 'keywords1'), 
                                                             (keywords2, st.session_state.lbl2, 'keywords2'), 
                                                             (keywords3, st.session_state.lbl3, 'keywords3')]):
                    if kws:
                        target_label = lbl_val.strip() if lbl_val.strip() != '' else fallback
                        df[target_label] = df['SelectedColumn'].apply(lambda x: 1 if wholeword(x, kws) else 0)
                        all_labels.append(target_label)
                
                df_summed = df.groupby('Date')[all_labels].sum().reset_index()
                st.session_state.df_graph_to_save = df_summed
                
                fig, ax = plt.subplots(figsize=(12, 5))
                for label in all_labels:
                    ax.plot(df_summed['Date'], df_summed[label], marker='o', label=label)
                    
                ax.set_ylabel('Frequency')
                ax.set_xlabel('Date')
                ax.legend()
                plt.xticks(rotation=30)
                st.pyplot(fig)
                
                st.dataframe(df_summed, use_container_width=True)
                
                # Active CSV layout structural memory download buffer export block
                csv_buffer = df_summed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Compiled Trend Calculations Output Frame (CSV)",
                    data=csv_buffer,
                    file_name="trend_comparison_output.csv",
                    mime="text/csv"
                )

# ------------------------------------------
# TAB 5: N-GRAM MATRIX EXTRAPOLATION
# ------------------------------------------
with tabs[4]:
    st.header("🔠 Textual N-Gram Distributions Metric Sets")
    if not st.session_state.processing_done:
        st.info("Execute 'Select Text' matrix compilation above to unlock this space.")
    else:
        nc1, nc2 = st.columns(2)
        with nc1:
            st.subheader("Top Extracted Unigrams")
            st.dataframe(st.session_state.unigrams, use_container_width=True)
        with nc2:
            st.subheader("Top Extracted Bigrams")
            st.dataframe(st.session_state.bigrams, use_container_width=True)

# ------------------------------------------
# TAB 6: NETWORK ADJACENCIES CO-OCCURRENCE
# ------------------------------------------
with tabs[5]:
    st.header("🕸️ Graph Structure Co-occurrence Matrix Maps")
    if not st.session_state.processing_done:
        st.info("Execute 'Select Text' matrix compilation above to unlock this space.")
    else:
        co_target = st.text_input("Focal Token Target Element Root Anchor", value=st.session_state.search_term if st.session_state.search_term else "text")
        
        if co_target:
            t_clean = co_target.lower().strip()
            next_nodes = st.session_state.word_freq_dict.get(t_clean, [])
            prev_nodes = st.session_state.prev_word_freq_dict.get(t_clean, [])
            all_neighbors = next_nodes + prev_nodes
            
            if all_neighbors:
                neighbor_counts = Counter(all_neighbors)
                edge_df = pd.DataFrame(neighbor_counts.items(), columns=['Linked Co-occurrence Token Node', 'Connection Linkage Frequency Weight'])
                edge_df = edge_df.sort_values(by='Connection Linkage Frequency Weight', ascending=False).reset_index(drop=True)
                
                st.subheader(f"Adjacency Structural Target Results Mapping for: '{t_clean}'")
                st.dataframe(edge_df, use_container_width=True)
                
                csv_buffer = edge_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"💾 Download Relational Graph Edge Paths List Matrix for '{t_clean}' (CSV)",
                    data=csv_buffer,
                    file_name=f"edges_{t_clean}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No direct adjacency token trajectories map to: '{t_clean}'")

# ==========================================
# 6. Application Structural Footer
# ==========================================
st.markdown("---")
st.caption("Copyright ©2026 Ikbal Maulana • Unified Content Analyzer Pipeline System Engine")