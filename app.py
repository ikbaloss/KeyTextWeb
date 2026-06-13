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
if "original_filename" not in st.session_state:
    st.session_state.original_filename = "keytext_dataset"

# Dictionary mappings for computing token weights
if "raw_tokens_list" not in st.session_state:
    st.session_state.raw_tokens_list = []
if "clean_sentences" not in st.session_state:
    st.session_state.clean_sentences = []

# Persistent variables for input components across redraws
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

# Helper function to re-compile model variables after data modification
def recompile_pipeline_matrices():
    df = st.session_state.main_data.copy()
    comments = [s for s in df['SelectedColumn'].to_list() if isinstance(s, str) and s.strip() != '']
    cleaned_comments = [keep_alphanumeric(s) for s in comments]
    
    st.session_state.clean_sentences = [s.split() for s in cleaned_comments]
    st.session_state.raw_tokens_list = re.findall(r'\b\w+(?:[-_]\w+)*\b', " ".join(comments))
    
    st.session_state.wv_model = Word2Vec(sentences=st.session_state.clean_sentences, min_count=20, vector_size=200, window=3, sg=1)
    st.session_state.main_data = df

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
                # Record the base name of the first primary file uploaded to use on export
                st.session_state.original_filename = os.path.splitext(uploaded_files[0].name)[0]
                
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

if st.button("⚡ Select Text & Compile Spatial Embeddings", type="primary"):
    if text_col_choice == "Select":
        st.error("Please identify a column header pointing to target language string text matrices.")
    else:
        with st.spinner("Extracting tokens and creating Word2Vec embeddings space..."):
            df = st.session_state.main_data.copy()
            df['SelectedColumn'] = df[text_col_choice].astype(str).str.lower()
            
            if date_col_choice != 'Select':
                df = df.rename(columns={date_col_choice: 'Date'})
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=day_first_chk, errors='coerce').dt.date
            
            df['SelectedColumn'] = df['SelectedColumn'].fillna('')
            
            stopword_file = "stopwords-id.txt" if lang_choice == "Indonesia" else "stopwords-en.txt"
            if os.path.exists(stopword_file):
                with open(stopword_file, "r") as tf:
                    st.session_state.stop_words = tf.read().split()
            else:
                st.session_state.stop_words = []
                
            st.session_state.main_data = df
            recompile_pipeline_matrices()
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

# TAB 1: DATASET OVERVIEW (WITH EXPORT MODIFIED DATASET CONTROL PANEL)
with tabs[0]:
    if st.session_state.main_data.empty:
        st.info("Awaiting structural data input files from sidebar control array panel loading.")
    else:
        # Dynamic filename builder appending current calendar day date pattern
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        export_filename = f"{st.session_state.original_filename}_{current_date_str}.csv"
        
        st.subheader("💾 Export Current Dataset Workspace")
        st.write("If you have run any keyword replacements or word normalization inside the vector or KWIC tabs, you can download the updated workspace here:")
        
        # Strip out working helper column if present before final download presentation
        df_to_export = st.session_state.main_data.copy()
        csv_buffer = df_to_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label=f"📥 Download Modified Dataset as CSV (`{export_filename}`)",
            data=csv_buffer,
            file_name=export_filename,
            mime="text/csv",
            type="secondary"
        )
        st.markdown("---")
        
        st.subheader("📋 Active Dataset View Matrix")
        st.dataframe(st.session_state.main_data, width="stretch")

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
                sim_df = pd.DataFrame(sims, columns=['Vocabulary Term Link', 'Cosine Distance Score Metric'])
                
                st.subheader(f"Vector Space Neighbor Results: '{t_clean}'")
                
                selection_event = st.dataframe(
                    sim_df, 
                    on_select="rerun", 
                    selection_mode="multi-row", 
                    width="stretch"
                )
                
                selected_indices = []
                if selection_event and hasattr(selection_event, "selection"):
                    sel = selection_event.selection
                    if isinstance(sel, dict):
                        selected_indices = sel.get("rows", [])
                    elif hasattr(sel, "rows"):
                        selected_indices = sel.rows
                elif isinstance(selection_event, dict) and "selection" in selection_event:
                    selected_indices = selection_event["selection"].get("rows", [])
                
                if selected_indices:
                    selected_words = sim_df.iloc[selected_indices]['Vocabulary Term Link'].tolist()
                    st.write(f"**Selected Terms for Substitution:** {', '.join(selected_words)}")
                    
                    if st.button(f"🔄 Replace Selected Terms with '{t_clean}'", type="primary", key="w2v_replace_btn"):
                        df = st.session_state.main_data.copy()
                        pattern = r'\b(?:{})\b'.format('|'.join([re.escape(w) for w in selected_words]))
                        df['SelectedColumn'] = df['SelectedColumn'].astype(str).str.replace(pattern, t_clean, regex=True)
                        
                        with st.spinner("Modifying underlying text matrices and re-compiling embeddings..."):
                            st.session_state.main_data = df
                            recompile_pipeline_matrices()
                            
                        st.success(f"Successfully replaced {', '.join(selected_words)} with '{t_clean}' across data columns!")
                        st.rerun()
                else:
                    st.info("💡 Select one or more rows via row checkboxes to activate the 'Replace' keyword feature.")
            else:
                st.error(f"Token vector mapping element '{t_clean}' not present in model vocabulary constraints.")

# TAB 3: KEY WORD IN CONTEXT (KWIC) EXPLORER (WITH PHRASES & WILDCARDS)
with tabs[2]:
    st.header("📖 Contextual Word Frame Windows")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to extract local contexts.")
    else:
        kwic_input_raw = st.text_input("Context Focus Targeted Node String (supports phrases and wildcards, e.g., merah putih, ber*, *nya)", value=st.session_state.kwic_term)
        
        cleaned_kwic_input = kwic_input_raw.strip().lower()
        st.session_state.kwic_term = cleaned_kwic_input

        kc1, kc2 = st.columns([2, 2])
        with kc1:
            w_radius = st.number_input("Token Extraction Window Slicing Horizon Radius (Words)", min_value=2, max_value=15, value=5)
        with kc2:
            replacement_word_input = st.text_input("Substitution Word / Target Token String (Optional Replacement)", value="")

        if cleaned_kwic_input:
            if "*" in cleaned_kwic_input:
                regex_parts = [re.escape(part) for part in cleaned_kwic_input.split("*")]
                kwic_pattern_string = r"\b" + r"\w*".join(regex_parts) + r"\b"
            else:
                kwic_pattern_string = r"\b" + re.escape(cleaned_kwic_input) + r"\b"
                
            kwic_compiled_regex = re.compile(kwic_pattern_string, re.IGNORECASE)

            if replacement_word_input.strip():
                rep_word_clean = replacement_word_input.strip().lower()
                
                if st.button(f"🔄 Replace all matching occurrences of '{cleaned_kwic_input}' with '{rep_word_clean}'", type="primary"):
                    df = st.session_state.main_data.copy()
                    df['SelectedColumn'] = df['SelectedColumn'].astype(str).apply(lambda x: kwic_compiled_regex.sub(rep_word_clean, x))
                    
                    with st.spinner("Executing structural context replacement and re-indexing corpora..."):
                        st.session_state.main_data = df
                        recompile_pipeline_matrices()
                        st.session_state.kwic_term = rep_word_clean
                        
                    st.success(f"Successfully replaced context targets with '{rep_word_clean}' across tracking arrays!")
                    st.rerun()

            records = []
            for raw_text in st.session_state.main_data['SelectedColumn'].dropna().astype(str):
                all_tokens = re.findall(r'\b\w+(?:[-_]\w+)*\b', raw_text.lower())
                
                for match in kwic_compiled_regex.finditer(raw_text.lower()):
                    matched_str = match.group()
                    matched_words = re.findall(r'\b\w+(?:[-_]\w+)*\b', matched_str)
                    if not matched_words:
                        continue
                        
                    for idx in range(len(all_tokens) - len(matched_words) + 1):
                        if all_tokens[idx:idx+len(matched_words)] == matched_words:
                            left_bound = max(0, idx - w_radius)
                            right_bound = min(len(all_tokens), idx + len(matched_words) + w_radius)
                            
                            records.append({
                                "Matched Term/Phrase": matched_str,
                                "Left Context Frame Segment": " ".join(all_tokens[left_bound:idx]),
                                "TARGET NODE INDEX": " ".join(all_tokens[idx:idx+len(matched_words)]),
                                "Right Context Frame Segment": " ".join(all_tokens[idx+len(matched_words):right_bound])
                            })
                            break
                            
            if records:
                st.subheader(f"Active Context Windows Frame for: '{cleaned_kwic_input}'")
                st.dataframe(pd.DataFrame(records), width="stretch")
            else:
                st.warning(f"No active window matching trajectories found for: '{cleaned_kwic_input}'")

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

        if st.button("📊 Construct Longitudinal Trend Curve Matrix Graphs", type="primary"):
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
                st.dataframe(df_summed, width="stretch")

# TAB 5: UPGRADED EXTRACTION N-GRAMS ENGINE
with tabs[4]:
    st.header("🔠 Complex N-Gram Frequency Extrapolation Core Matrix")
    if not st.session_state.processing_done:
        st.info("Run core parameter execution above to populate multi-gram configurations tables.")
    else:
        gram_selection = st.radio(
            "Select Target N-Gram Analysis Dimension Depth Scope", 
            options=["Unigram (1-Gram)", "Bigram (2-Gram)", "Trigram (3-Gram)", "4-Gram", "5-Gram"],
            horizontal=True
        )
        
        n_mapping = {"Unigram (1-Gram)": 1, "Bigram (2-Gram)": 2, "Trigram (3-Gram)": 3, "4-Gram": 4, "5-Gram": 5}
        target_n_val = n_mapping[gram_selection]
        
        with st.spinner(f"Compiling frequency distributions for chosen structural sequence segment: {gram_selection}..."):
            ngram_dataframe_output = get_ngram_dataframe(st.session_state.clean_sentences, target_n_val, st.session_state.stop_words)
            
        st.subheader(f"Ranked Distribution Matrix View Output: {gram_selection}")
        st.dataframe(ngram_dataframe_output, width="stretch")
        
        st.download_button(
            label=f"💾 Download {gram_selection} Frequency Balance Matrix Sheet (CSV)",
            data=ngram_dataframe_output.to_csv(index=False).encode('utf-8'),
            file_name=f"frequency_distribution_{target_n_val}gram.csv",
            mime="text/csv"
        )

# TAB 6: ADVANCED NETWORK CO-OCCURRENCE MODELER
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
            
        if st.button("🔮 Calculate Adjacency Co-occurrences & Compile Graph Weights Matrix"):
            with st.spinner("Extracting structural graph edges, logDice indices, and semantic similarity mappings..."):
                
                flat_words = [w for s in st.session_state.clean_sentences for w in s if w not in st.session_state.stop_words]
                global_counts_dict = Counter(flat_words)
                
                top_unigrams_list = global_counts_dict.most_common(top_unigram_limit)
                target_nodes_set = set([item[0] for item in top_unigrams_list])
                
                cooccur_counts_matrix = defaultdict(int)
                
                for sentence in st.session_state.clean_sentences:
                    filtered_s = [w for w in sentence if w in target_nodes_set]
                    s_len = len(filtered_s)
                    for i in range(s_len):
                        w1 = filtered_s[i]
                        for j in range(i + 1, min(s_len, i + co_window_span + 1)):
                            w2 = filtered_s[j]
                            if w1 != w2:
                                sorted_pair = tuple(sorted([w1, w2]))
                                cooccur_counts_matrix[sorted_pair] += 1
                                
                    for w in set(filtered_s):
                        if filtered_s.count(w) > 1:
                            cooccur_counts_matrix[(w, w)] += (filtered_s.count(w) - 1)
                                
                network_rows_records = []
                for (node_a, node_b), co_freq in cooccur_counts_matrix.items():
                    if co_freq > 0:
                        freq_a = global_counts_dict.get(node_a, 1)
                        freq_b = global_counts_dict.get(node_b, 1)
                        
                        dice_inner_val = (2 * co_freq) / (freq_a + freq_b)
                        log_dice_value = 14 + math.log2(dice_inner_val) if dice_inner_val > 0 else 0.0
                        
                        cosine_sim_value = 0.0
                        if node_a == node_b:
                            cosine_sim_value = 1.0
                        elif st.session_state.wv_model and node_a in st.session_state.wv_model.wv and node_b in st.session_state.wv_model.wv:
                            cosine_sim_value = float(st.session_state.wv_model.wv.similarity(node_a, node_b))
                            
                        network_rows_records.append({
                            "Source": node_a,
                            "Target": node_b,
                            "frequency": co_freq,
                            "association": round(log_dice_value, 4),
                            "similarity": round(cosine_sim_value, 4)
                        })
                        
                if network_rows_records:
                    compiled_edges_dataframe = pd.DataFrame(network_rows_records)
                    
                    st.success(f"Assembled relational network containing {len(compiled_edges_dataframe)} unique graph edge pathways.")
                    st.dataframe(compiled_edges_dataframe, width="stretch")
                    
                    gml_lines_list = ["graph [", "  directed 0"]
                    all_unique_nodes = [item[0] for item in top_unigrams_list]
                    node_to_id_index_map = {node_str: idx for idx, node_str in enumerate(all_unique_nodes)}
                    
                    for n_str in all_unique_nodes:
                        abs_freq = global_counts_dict.get(n_str, 0)
                        gml_lines_list.append("  node [")
                        gml_lines_list.append(f'    id {node_to_id_index_map[n_str]}')
                        gml_lines_list.append(f'    label "{n_str}"')
                        gml_lines_list.append(f'    frequency {abs_freq}')
                        gml_lines_list.append("  ]")
                        
                    for edge_record in network_rows_records:
                        if edge_record["Source"] in node_to_id_index_map and edge_record["Target"] in node_to_id_index_map:
                            id_source = node_to_id_index_map[edge_record["Source"]]
                            id_target = node_to_id_index_map[edge_record["Target"]]
                            
                            gml_lines_list.append("  edge [")
                            gml_lines_list.append(f"    source {id_source}")
                            gml_lines_list.append(f"    target {id_target}")
                            gml_lines_list.append(f"    frequency {edge_record['frequency']}")
                            gml_lines_list.append(f"    association {edge_record['association']}")
                            gml_lines_list.append(f"    similarity {edge_record['similarity']}")
                            gml_lines_list.append("  ]")
                        
                    gml_lines_list.append("]")
                    final_gml_string_output = "\n".join(gml_lines_list)
                    
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.download_button(
                            label="🕸️ Export Structural Network Mapping File (.gml)",
                            data=final_gml_string_output,
                            file_name="keytext_graph_network.gml",
                            mime="text/plain",
                            key="gml_download_btn"
                        )
                    with d_col2:
                        st.download_button(
                            label="📊 Download Edge Weights Relational Spreadsheet (.csv)",
                            data=compiled_edges_dataframe.to_csv(index=False).encode('utf-8'),
                            file_name="keytext_network_edges.csv",
                            mime="text/csv",
                            key="csv_download_btn"
                        )
                else:
                    st.warning("No co-occurrence connections found matching your settings thresholds.")

# ==========================================
# 6. Structural Theme Application Footer
# ==========================================
st.markdown("---")
st.caption("Copyright ©2026 Ikbal Maulana • Unified Content Analyzer Pipeline System Engine")