import os
import re
from datetime import datetime, timedelta
import pandas as pd
import collections
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gensim.models import Word2Vec
import streamlit as st
import io  # Ensure this import is added at the top of your app.py file

# ==========================================
# 1. Page Config & Session State Init
# ==========================================
st.set_page_config(page_title="KeyText Version 0.22", layout="wide")
st.title("🔑 KeyText Version 0.22")

# Replicating App() class level variables using Streamlit Session State
if "main_data" not in st.session_state:
    st.session_state.main_data = pd.DataFrame()
if "unigrams" not in st.session_state:
    st.session_state.unigrams = pd.DataFrame()
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

# State variables for Category Comparison inputs
for key in ["kw1", "kw2", "kw3", "lbl1", "lbl2", "lbl3"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# ==========================================
# 2. Utility & Analysis Helper Functions
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
        unigram_freq = Counter(tokens)
        unigrams_df = pd.DataFrame(unigram_freq.items(), columns=['Unigram', 'Frequency'])
        return unigrams_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return pd.DataFrame()

# ==========================================
# 3. Sidebar Actions (Simulating Menubar/File Operations)
# ==========================================
with st.sidebar:
    st.header("📁 File Actions")
    uploaded_files = st.file_uploader("Open CSV or TXT Files", accept_multiple_files=True, type=["csv", "txt"])
    
    # Custom delimiter field for TXT parsing (replaces QInputDialog)
    txt_delimiter = st.text_input("Paragraph Delimiter (for TXT only)", value="\\n")
    txt_delimiter = txt_delimiter.replace('\\n', '\n')

    if st.button("Process & Load Files"):
        if not uploaded_files:
            st.error("There is no file to upload!")
        else:
            list_of_files = []
            file_types = [f.name.split('.')[-1] for f in uploaded_files]
            
            if len(set(file_types)) > 1:
                st.error("All files must be of the same type, csv or txt")
            else:
                is_csv = file_types[0] == 'csv'
                
                

                # ... inside your sidebar file loading loop ...

                for idx, f in enumerate(uploaded_files):
                    fileName = os.path.splitext(f.name)[0]
                    if is_csv:
                        # Wrap the stream in a TextIOWrapper to handle the backslash replacement safely
                        text_stream = io.TextIOWrapper(f, encoding='utf-8', errors='backslashreplace')
                        dfcsv = pd.read_csv(text_stream)
                        
                        if len(uploaded_files) > 1:
                            dfcsv.insert(loc=1, column='Data', value=[fileName]*len(dfcsv))
                        list_of_files.append(dfcsv)
                    else:
                        # For text files, read and decode the raw bytes manually
                        raw_bytes = f.read()
                        raw_text = raw_bytes.decode('utf-8', errors='backslashreplace')
                        paragraphs = raw_text.split(txt_delimiter)
                        
                        # ... rest of your original paragraph processing logic ...
                        
                        for i in range(n_dates):
                            current_group_size = group_size + 1 if i < remainder else group_size
                            date_assignments.extend([date_idx] * min(current_group_size, n_rows - len(date_assignments)))
                            date_idx += 1
                            if len(date_assignments) >= n_rows:
                                break
                        
                        all_dates = [datetime.now().date() - timedelta(days=(n_dates - 1 - k)) for k in range(n_dates)]
                        dftext['Date'] = [all_dates[idx] for idx in date_assignments]
                        list_of_files.append(dftext)
                
                df_combined = pd.concat(list_of_files, ignore_index=True)
                df_combined.drop_duplicates(inplace=True)
                st.session_state.main_data = df_combined
                st.success("Files parsed and merged successfully!")

    # Export Logic (replaces QFileDialog.getSaveFileName)
    if not st.session_state.main_data.empty:
        st.write("---")
        st.subheader("💾 Export Data")
        csv_data = st.session_state.main_data.to_csv(date_format='%Y.%m.%d', encoding='utf-8', index=False)
        st.download_button(
            label="Save Main Data to CSV",
            data=csv_data,
            file_name="keytext_output.csv",
            mime="text/csv"
        )

# ==========================================
# 4. Main Tab Interface
# ==========================================
# Construct tabs similar to self.tabs.addTab
tabs = st.tabs(["Raw Data", "Search KeyWord", "KWIC", "Category Comparison", "N-Gram", "Cooccurence"])

# ------------------------------------------
# TAB 1: RAW DATA
# ------------------------------------------
with tabs[0]:
    st.header("📊 Raw Data Configurations")
    
    if st.session_state.main_data.empty:
        st.info("Please open and process files using the sidebar configuration first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            lang_choice = st.selectbox("Language", options=["Indonesia", "English"])
        with col2:
            columns_list = list(st.session_state.main_data.columns)
            date_col_choice = st.selectbox("Date Column", options=["Select"] + columns_list, index=columns_list.index("Date") + 1 if "Date" in columns_list else 0)
        with col3:
            day_first_chk = st.checkbox("Day First", value=False)
        with col4:
            text_col_choice = st.selectbox("Text Column", options=["Select"] + columns_list, index=columns_list.index("SelectedColumn") + 1 if "SelectedColumn" in columns_list else 0)
            
        if st.button("⚡ Select & Process Text (NLP Initialization)", type="primary"):
            if text_col_choice == "Select":
                st.warning("Please select at least one item from the Text Column field!")
            else:
                df = st.session_state.main_data.copy()
                
                # Setup SelectedColumn
                if text_col_choice != 'SelectedColumn':
                    if 'SelectedColumn' in df.columns:
                        df = df.drop('SelectedColumn', axis=1)
                    df['SelectedColumn'] = df[text_col_choice].astype(str).str.lower()
                
                # Parse Date formats
                if date_col_choice != 'Select':
                    df = df.rename(columns={date_col_choice: 'Date'})
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=day_first_chk, errors='coerce').dt.date
                
                df['SelectedColumn'] = df['SelectedColumn'].fillna('')
                comments = [s for s in df['SelectedColumn'].to_list() if isinstance(s, str) and s.strip() != '']
                cleaned_comments = [keep_alphanumeric(s) for s in comments]
                token_comments = [s.split() for s in cleaned_comments]
                
                # Train Word2Vec Model
                with st.spinner("Training Word2Vec Model..."):
                    wv_model = Word2Vec(
                        sentences=token_comments,
                        min_count=20,
                        vector_size=200,
                        window=3,
                        compute_loss=True,
                        sg=1
                    )
                
                # Load Stopwords lists
                stopword_file = "stopwords-id.txt" if lang_choice == "Indonesia" else "stopwords-en.txt"
                if os.path.exists(stopword_file):
                    with open(stopword_file, "r") as tf:
                        st.session_state.stop_words = tf.read().split()
                else:
                    st.session_state.stop_words = []
                
                # Compute Next/Previous word contextual alignments
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
                st.session_state.main_data = df
                st.session_state.processing_done = True
                st.balloons()
                
        st.subheader("Data Overview Table")
        st.dataframe(st.session_state.main_data, use_container_width=True)

# ------------------------------------------
# TAB 2 & 3: STUBS FOR OTHER CORE PHASES
# ------------------------------------------
with tabs[1]:
    st.header("🔍 Search Keyword Space")
    if not st.session_state.processing_done:
        st.info("Execute 'Select & Process Text' in Tab 1 to activate processing modules.")
    else:
        st.write("Word2Vec model is loaded. Ready to pull vocabulary attributes.")
        # You can add model vector similarity search terms here.

with tabs[2]:
    st.header("📖 Key Word in Context (KWIC)")
    if not st.session_state.processing_done:
        st.info("Execute 'Select & Process Text' in Tab 1 to activate processing modules.")

# ------------------------------------------
# TAB 4: CATEGORY COMPARISON
# ------------------------------------------
with tabs[3]:
    st.header("📈 Category Comparison & Trends")
    if not st.session_state.processing_done:
        st.info("Execute 'Select & Process Text' in Tab 1 to activate processing modules.")
    else:
        # Layout matching self.glComparison layout grid items
        r1c1, r1c2, r1c3, r1c4 = st.columns([3, 2, 1, 1])
        with r1c1: st.session_state.kw1 = st.text_input("Keywords Group 1 (separated by | )", value=st.session_state.kw1)
        with r1c2: st.session_state.lbl1 = st.text_input("Label As Group 1", value=st.session_state.lbl1)
        with r1c3: 
            if st.button("🧹 Clear G1"): st.session_state.kw1, st.session_state.lbl1 = "", ""
            
        r2c1, r2c2, r2c3, r2c4 = st.columns([3, 2, 1, 1])
        with r2c1: st.session_state.kw2 = st.text_input("Keywords Group 2 (separated by | )", value=st.session_state.kw2)
        with r2c2: st.session_state.lbl2 = st.text_input("Label As Group 2", value=st.session_state.lbl2)
        with r2c3:
            if st.button("🧹 Clear G2"): st.session_state.kw2, st.session_state.lbl2 = "", ""

        r3c1, r3c2, r3c3, r3c4 = st.columns([3, 2, 1, 1])
        with r3c1: st.session_state.kw3 = st.text_input("Keywords Group 3 (separated by | )", value=st.session_state.kw3)
        with r3c2: st.session_state.lbl3 = st.text_input("Label As Group 3", value=st.session_state.lbl3)
        with r3c3:
            if st.button("🧹 Clear G3"): st.session_state.kw3, st.session_state.lbl3 = "", ""

        # Chart Action Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            side_by_side = st.button("📊 Plot Side by Side Timeline")
        with btn_col2:
            filtered_comp = st.button("🔍 Plot Filtered Comparison")

        if side_by_side or filtered_comp:
            df = st.session_state.main_data.copy()
            keywords1 = [w.strip() for w in st.session_state.kw1.split('|') if w.strip() != '']
            keywords2 = [w.strip() for w in st.session_state.kw2.split('|') if w.strip() != '']
            keywords3 = [w.strip() for w in st.session_state.kw3.split('|') if w.strip() != '']
            
            if not (keywords1 or keywords2 or keywords3):
                st.warning("Please provide keywords to generate the chart!")
            else:
                all_labels = []
                # Map keyword clusters to evaluation rows
                for i, (kws, lbl_val, fallback) in enumerate([(keywords1, st.session_state.lbl1, 'keywords1'), 
                                                             (keywords2, st.session_state.lbl2, 'keywords2'), 
                                                             (keywords3, st.session_state.lbl3, 'keywords3')]):
                    if kws:
                        target_label = lbl_val.strip() if lbl_val.strip() != '' else fallback
                        df[target_label] = df['SelectedColumn'].apply(lambda x: 1 if wholeword(x, kws) else 0)
                        all_labels.append(target_label)
                
                # Perform Time aggregations
                df = df[['Date'] + all_labels]
                df_summed = df.groupby('Date').sum().reset_index()
                
                start_date = df_summed['Date'].min()
                end_date = df_summed['Date'].max()
                
                # Matplotlib Canvas configuration context
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_xlim(start_date, end_date)
                
                num_dates = len(pd.date_range(start=start_date, end=end_date))
                n = max(1, num_dates // 10)
                
                for label in all_labels:
                    ax.plot(df_summed['Date'], df_summed[label], label=label)
                    
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=n))
                ax.set_ylabel('Frequency')
                ax.set_xlabel('Date')
                ax.legend()
                plt.xticks(rotation=30)
                
                # Display output explicitly via streamlit components
                st.pyplot(fig)
                
                # Allow table extraction 
                st.subheader("Chart Data Stream Source")
                st.dataframe(df_summed)
                st.download_button(label="Export Summed Trend Matrix (.csv)", data=df_summed.to_csv(), file_name="trends.csv")

# ------------------------------------------
# TAB 5: N-GRAMS
# ------------------------------------------
with tabs[4]:
    st.header("🔠 N-Gram Extractor Data View")
    if not st.session_state.processing_done:
        st.info("Execute 'Select & Process Text' in Tab 1 to activate processing modules.")
    else:
        st.write("Unigrams processed automatically upon loading:")
        st.dataframe(st.session_state.unigrams, use_container_width=True)

# ------------------------------------------
# TAB 6: COOCCURENCE
# ------------------------------------------
with tabs[5]:
    st.header("🕸️ Network Cooccurence Maps")
    if not st.session_state.processing_done:
        st.info("Execute 'Select & Process Text' in Tab 1 to activate processing modules.")

# ==========================================
# 5. Footer Layout Attributes
# ==========================================
st.markdown("---")
st.caption("Copyright ©2026 Ikbal Maulana • Migrated seamlessly to Streamlit Cloud Web Environment")
