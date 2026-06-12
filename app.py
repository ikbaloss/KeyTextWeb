import os
import re
import io
import math
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
# 1. Page Configuration & State Matrix
# ==========================================
st.set_page_config(page_title="KeyText Version 0.22", layout="wide")
st.title("🔑 KeyText Version 0.22")

# Core Memory Data Engine States
if "main_data" not in st.session_state:
    st.session_state.main_data = pd.DataFrame()
if "stop_words" not in st.session_state:
    st.session_state.stop_words = []
if "wv_model" not in st.session_state:
    st.session_state.wv_model = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# Dictionary mappings for computing positional weights
if "raw_tokens_list" not in st.session_state:
    st.session_state.raw_tokens_list = []
if "clean_sentences" not in st.session_state:
    st.session_state.clean_sentences = []

# Persistent variables for user inputs across redraws
for key in ["kw1", "kw2", "kw3", "lbl1", "lbl2", "lbl3", "search_term", "kwic_term"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ==========================================
# 2. Mathematical & Tokenizer Help Matrix
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

def get_ngram_dataframe(sentences, n, stop_words):
    ngram_list = []
    for words in sentences:
        filtered = [w for w in words if w not in stop_words]
        if len(filtered) >= n:
            for i in range(len(filtered) - n + 1):
                ngram_list.append(" ".join(filtered[i:i+n]))
    
    counts = Counter(ngram_list)
    df_ngram = pd.DataFrame(counts.items(), columns=[f'{n}-Gram', 'Frequency'])
    return df_ngram.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

# ==========================================
# 3. Sidebar Configuration Loader
# ==========================================
with st.sidebar:
    st.header("📁 File Upload System")
    uploaded_files = st.file_uploader("Open CSV or TXT Files", accept_multiple_files=True, type=["csv", "txt"])
    
    txt_delimiter = st.text_input("Paragraph Delimiter (for TXT files)", value="\\n")
    txt_delimiter = txt_delimiter.replace('\\n', '\n')

    if st.button("🚀 Load Uploaded Files", use_container_width=True):
        if not uploaded_files:
            st.error("Please provide valid source target data files.")
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
                st.session_state.processing_done = False
                st.success(f"Assembled {len(df_combined)} unique workspace records.")

# ==========================================
# 4. Global Target Parameter Dropdowns
# ==========================================
columns_list = ["Select"] + list(st.session_state.main_data.columns) if not st.session_state.main_data.empty else ["Select"]

st.subheader("⚙️ Analysis Pipeline Settings Panel")
col1, col2, col3, col4 = st.columns(4)

with col1:
    lang_choice = st.selectbox("Language Selection", options=["Indonesia", "English"])
with col2:
    date_col_choice = st.selectbox("Date Identifier Header", options=columns_list, index=columns_list.index("Date") if "Date" in columns_list else 0)
with col3:
    day_first_chk = st.checkbox("Day First Datetime Flag", value=False)
with col4:
    text_col_choice = st.selectbox("Linguistic Text Column Target", options=columns_list, index=columns_list.index("Text") if "Text" in columns_list else 0)

if st.button("⚡ Select Text & Compile Spatial Embeddings", type="primary", icon="⚡"):
    if text_col_choice == "Select":
        st.error("Please identify a column header pointing to target language string text matrices.")
    else:
        # Fixed lowercase st.spinner typo below:
        with st.spinner("Extracting tokens and creating Word2Vec embeddings space..."):
            df = st.session_state.main_data.copy()
            df['SelectedColumn'] = df[text_col_choice].astype(str).str.lower()
            
            if date_col_choice != 'Select':
                df = df.rename(columns={date_col_choice: 'Date'})
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=day_first_chk, errors='coerce').dt.date
            
            df['SelectedColumn'] = df['SelectedColumn'].fillna('')
            comments = [s for s in df['SelectedColumn'].to_list() if isinstance(s, str) and s.strip() != '']
            cleaned_comments = [keep_alphanumeric(s) for s in comments]
            
            st.session_state.clean_sentences = [s.split() for s in cleaned_comments]
            st.session_state.raw_tokens_list = re.findall(r'\b\w+(?:[-_]\w+)*\b', " ".join(comments))
            
            stopword_file = "stopwords-id.txt" if lang_choice == "Indonesia" else "stopwords-en.txt"
            if os.path.exists(stopword_file):
                with open(stopword_file, "r") as tf:
                    st.session_state.stop_words = tf.read().split()
            else:
                st.session_state.stop_words = []
            
            st.session_state.wv_model = Word2Vec(sentences=st.session_state.clean_sentences, min_count=20, vector_size=200, window=3, sg=1)
            st.session_state.main_data = df
            st.session_state.processing_done = True
            st.success("Linguistic embedding configuration maps computed successfully!")

# ==========================================
# 5. Modular Workspace Environment Tabs
# ==========================================
tabs = st.tabs([
    "📊 Dataset Overview", 
    "🔍 Word2Vec Vector Space", 
    "📖 KWIC Slicer", 
    "📈 Trend Visualizations", 
    "🔠 N-Gram Distributions", 
    "🕸️ Co-occurrence Modeler"
])

# TAB 1: DATASET OVERVIEW
with tabs[0]:
    if st.session_state.main_data.empty:
        st.info("Awaiting structural document matrices data input files from sidebar control array panel loading.")
    else:
        st.dataframe(st.session_state.main_data, use_container_width=True)

# TAB 2: WORD2VEC SEMANTIC MAPS
with tabs[1]:
    st.header("🔍 Neural Embedding Distance Vector Maps")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to load word embeddings mapping vectors.")
    else:
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            st.session_state.search_term = st.text_input("Target Model Vocabulary Word Token", value=st.session_state.search_term)
        with sc2:
            top_n = st.number_input("Top Matches Upper Limit Bound", min_value=5, max_value=100, value=20)
            
        if st.session_state.search_term:
            t_clean = st.session_state.search_term.lower().strip()
            if st.session_state.wv_model and t_clean in st.session_state.wv_model.wv:
                sims = st.session_state.wv_model.wv.most_similar(t_clean, topn=top_n)
                st.dataframe(pd.DataFrame(sims, columns=['Vocabulary Term Link', 'Cosine Distance Score Metric']), use_container_width=True)
            else:
                st.error(f"Token vector mapping element '{t_clean}' not present in model vocabulary constraints.")

# TAB 3: KEY WORD IN CONTEXT (KWIC) EXPLORER
with tabs[2]:
    st.header("📖 Contextual Word Frame Windows")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to extract local contexts.")
    else:
        kc1, kc2 = st.columns([3, 1])
        with kc1:
            st.session_state.kwic_term = st.text_input("Context Focus Targeted Node String", value=st.session_state.kwic_term)
        with kc2:
            w_radius = st.number_input("Token Extraction Window Slicing Horizon Radius", min_value=2, max_value=15, value=5)
            
        if st.session_state.kwic_term:
            target_kwic = st.session_state.kwic_term.lower().strip()
            records = []
            for words in st.session_state.clean_sentences:
                for idx, tok in enumerate(words):
                    if tok == target_kwic:
                        records.append({
                            "Left Context Frame Segment": " ".join(words[max(0, idx - w_radius):idx]),
                            "TARGET NODE INDEX": words[idx],
                            "Right Context Frame Segment": " ".join(words[idx+1:min(len(words), idx + w_radius + 1)])
                        })
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True)
            else:
                st.warning("No overlapping window sequences surfaced along current tokens list data matrix.")

# TAB 4: TREND REVIEWS
with tabs[3]:
    st.header("📈 Frequency Series Trend Evaluations")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to activate statistical calculators.")
    else:
        g1c1, g1c2 = st.columns([3, 2])
        with g1c1: st.session_state.kw1 = st.text_input("Group 1 Expressions Array (pipe | syntax separate)", value=st.session_state.kw1)
        with g1c2: st.session_state.lbl1 = st.text_input("Group Label Header Map Title Alias 1", value=st.session_state.lbl1)
        
        g2c1, g2c2 = st.columns([3, 2])
        with g2c1: st.session_state.kw2 = st.text_input("Group 2 Expressions Array (pipe | syntax separate)", value=st.session_state.kw2)
        with g2c2: st.session_state.lbl2 = st.text_input("Group Label Header Map Title Alias 2", value=st.session_state.lbl2)

        g3c1, g3c2 = st.columns([3, 2])
        with g3c1: st.session_state.kw3 = st.text_input("Group 3 Expressions Array (pipe | syntax separate)", value=st.session_state.kw3)
        with g3c2: st.session_state.lbl3 = st.text_input("Group Label Header Map Title Alias 3", value=st.session_state.lbl3)

        if st.button("📊 Construct Longitudinal Trend Curve Matrix Graphs", type="primary", use_container_width=True):
            df = st.session_state.main_data.copy()
            k1 = [w.strip() for w in st.session_state.kw1.split('|') if w.strip() != '']
            k2 = [w.strip() for w in st.session_state.kw2.split('|') if w.strip() != '']
            k3 = [w.strip() for w in st.session_state.kw3.split('|') if w.strip() != '']
            
            if not (k1 or k2 or k3):
                st.warning("Trend comparison fields empty. Awaiting targeted strings compilation loops data matrix.")
            else:
                labels = []
                for idx, (kws, l_val, fb) in enumerate([(k1, st.session_state.lbl1, 'Group 1'), (k2, st.session_state.lbl2, 'Group 2'), (k3, st.session_state.lbl3, 'Group 3')]):
                    if kws:
                        name = l_val.strip() if l_val.strip() != '' else fb
                        df[name] = df['SelectedColumn'].apply(lambda x: 1 if wholeword(x, kws) else 0)
                        labels.append(name)
                
                df_summed = df.groupby('Date')[labels].sum().reset_index()
                fig, ax = plt.subplots(figsize=(12, 4))
                for lbl in labels:
                    ax.plot(df_summed['Date'], df_summed[lbl], marker='o', label=lbl)
                ax.legend()
                plt.xticks(rotation=30)
                st.pyplot(fig)
                st.dataframe(df_summed, use_container_width=True)

# ------------------------------------------
# TAB 5: UPGRADED EXTRACTION N-GRAMS ENGINE
# ------------------------------------------
with tabs[4]:
    st.header("🔠 Complex N-Gram Frequency Extrapolation Core Matrix")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to populate multi-gram configurations tables.")
    else:
        # Dynamic selection widget mapping requested for continuous spectrum validation bounds
        gram_selection = st.radio(
            "Select Target N-Gram Analysis Dimension Depth Scope", 
            options=["Unigram (1-Gram)", "Bigram (2-Gram)", "Trigram (3-Gram)", "4-Gram", "5-Gram"],
            horizontal=True
        )
        
        # Determine target N variable layer based on programmatic widget selection returns
        n_mapping = {"Unigram (1-Gram)": 1, "Bigram (2-Gram)": 2, "Trigram (3-Gram)": 3, "4-Gram": 4, "5-Gram": 5}
        target_n_val = n_mapping[gram_selection]
        
        with st.spinner(f"Compiling frequency distributions for chosen structural sequence segment: {gram_selection}..."):
            ngram_dataframe_output = get_ngram_dataframe(st.session_state.clean_sentences, target_n_val, st.session_state.stop_words)
            
        st.subheader(f"Ranked Distribution Matrix View Output: {gram_selection}")
        st.dataframe(ngram_dataframe_output, use_container_width=True)
        
        # Download handler for specific calculated segment sets matrix file extractions
        st.download_button(
            label=f"💾 Download {gram_selection} Frequency Balance Matrix Sheet (CSV)",
            data=ngram_dataframe_output.to_csv(index=False).encode('utf-8'),
            file_name=f"frequency_distribution_{target_n_val}gram.csv",
            mime="text/csv"
        )

# ------------------------------------------
# TAB 6: ADVANCED NETWORK CO-OCCURRENCE MODELER
# ------------------------------------------
with tabs[5]:
    st.header("🕸️ Relational Structural Token Co-occurrence Matrices Modeler")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to unlock complex relational calculations matrices.")
    else:
        st.write("Construct relational adjacency maps based on structural LogDice metrics:")
        
        cc1, cc2 = st.columns(2)
        with cc1:
            top_unigram_limit = st.number_input("Top N Unigram Filter Target Threshold Size", min_value=10, max_value=500, value=150)
        with cc2:
            co_window_span = st.number_input("Adjacency Context Token Span Window Range (Span Size)", min_value=2, max_value=10, value=4)
            
        if st.button("🔮 Calculate Adjacency Co-occurrences & Compile Graph Weights Matrix", type="primary", use_container_width=True):
            with st.spinner("Extracting structural graph edges, logDice indices, and semantic similarity mappings..."):
                
                # 1. Isolate the Top N Unigrams using clean word lists definitions filter loops
                flat_words = [w for s in st.session_state.clean_sentences for w in s if w not in st.session_state.stop_words]
                top_unigrams_counts = Counter(flat_words).most_common(top_unigram_limit)
                target_nodes_set = set([item[0] for item in top_unigrams_counts])
                freq_dict_nodes = dict(top_unigrams_counts)
                
                # Total baseline dictionary frequency units size for calculation checks
                total_tokens_volume = len(flat_words)
                
                # 2. Populate spatial transactional distance co-occurrences using span constraints
                cooccur_counts_matrix = defaultdict(int)
                
                for sentence in st.session_state.clean_sentences:
                    filtered_s = [w for w in sentence if w in target_nodes_set]
                    s_len = len(filtered_s)
                    for i in range(s_len):
                        w1 = filtered_s[i]
                        # Look forward within the designated calculation span limit boundaries window parameters
                        for j in range(i + 1, min(s_len, i + co_window_span + 1)):
                            w2 = filtered_s[j]
                            if w1 != w2:
                                # Ensure deterministic structural sort order keys matching original desktop layout assignments
                                sorted_pair = tuple(sorted([w1, w2]))
                                cooccur_counts_matrix[sorted_pair] += 1
                                
                # 3. Apply standard formulas to calculate weights, logDice, and semantic similarities
                network_rows_records = []
                for (node_a, node_b), co_freq in cooccur_counts_matrix.items():
                    if co_freq > 0:
                        freq_a = freq_dict_nodes.get(node_a, 1)
                        freq_b = freq_dict_nodes.get(node_b, 1)
                        
                        # Apply desktop logDice index formula: 14 + log2( (2 * Co) / (FreqA + FreqB) )
                        dice_inner_val = (2 * co_freq) / (freq_a + freq_b)
                        log_dice_value = 14 + math.log2(dice_inner_val) if dice_inner_val > 0 else 0.0
                        
                        # Extract similarity via Word2Vec spatial vector model weights matrix targets
                        cosine_sim_value = 0.0
                        if st.session_state.wv_model and node_a in st.session_state.wv_model.wv and node_b in st.session_state.wv_model.wv:
                            cosine_sim_value = float(st.session_state.wv_model.wv.similarity(node_a, node_b))
                            
                        network_rows_records.append({
                            "Source": node_a,
                            "Target": node_b,
                            "Weight": co_freq,
                            "LogDice": round(log_dice_value, 4),
                            "Similarity": round(cosine_sim_value, 4)
                        })
                        
                if network_rows_records:
                    compiled_edges_dataframe = pd.DataFrame(network_rows_records)
                    
                    st.success(f"Assembled relational network containing {len(compiled_edges_dataframe)} unique graph edge pathways.")
                    st.dataframe(compiled_edges_dataframe, use_container_width=True)
                    
                    # 4. Generate the exact GML file format matching your desktop text stream output requirements
                    gml_lines_list = ["graph [", "  directed 0"]
                    
                    # Track assigned string node targets mapping indexes
                    all_unique_nodes = sorted(list(target_nodes_set))
                    node_to_id_index_map = {node_str: idx for idx, node_str in enumerate(all_unique_nodes)}
                    
                    # Inject node declarations into GML text block
                    for n_str in all_unique_nodes:
                        gml_lines_list.append("  node [")
                        gml_lines_list.append(f'    id {node_to_id_index_map[n_str]}')
                        gml_lines_list.append(f'    label "{n_str}"')
                        gml_lines_list.append("  ]")
                        
                    # Inject edge connection records containing metadata fields directly into GML blocks
                    for edge_record in network_rows_records:
                        id_source = node_to_id_index_map[edge_record["Source"]]
                        id_target = node_to_id_index_map[edge_record["Target"]]
                        
                        gml_lines_list.append("  edge [")
                        gml_lines_list.append(f"    source {id_source}")
                        gml_lines_list.append(f"    target {id_target}")
                        gml_lines_list.append(f"    weight {edge_record['Weight']}")
                        gml_lines_list.append(f"    logDice {edge_record['LogDice']}")
                        gml_lines_list.append(f"    similarity {edge_record['Similarity']}")
                        gml_lines_list.append("  ]")
                        
                    gml_lines_list.append("]")
                    final_gml_string_output = "\n".join(gml_lines_list)
                    
                    # 5. Provide download access options for both data stream tracking formats
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.download_button(
                            label="🕸️ Export Structural Network Mapping File (.gml)",
                            data=final_gml_string_output,
                            file_name="keytext_graph_network.gml",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with d_col2:
                        st.download_button(
                            label="📊 Download Edge Weights Relational Spreadsheet (.csv)",
                            data=compiled_edges_dataframe.to_csv(index=False).encode('utf-8'),
                            file_name="keytext_network_edges.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.warning("No co-occurrence connections found matching your settings thresholds.")

# ==========================================
# 6. Structural Theme Application Footer
# ==========================================
st.markdown("---")
st.caption("Copyright ©2026 Ikbal Maulana • Unified Content Analyzer Pipeline System Engine")
