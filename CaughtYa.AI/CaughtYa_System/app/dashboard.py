import streamlit as st
import requests
import json
import os
import sys

# Define API URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="CaughtYa - Fake News Detector", layout="wide")

st.title("🕵️ CaughtYa - AI & Fake News Detection System")
st.markdown("Analyze news articles for authenticity, AI generation, and internal consistency.")

# Input area
input_text = st.text_area("Enter News / Article Text Here:", height=200)

if st.button("Analyze Content", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text... (This may take a moment)"):
            try:
                # Call FastAPI backend
                response = requests.post(API_URL, json={"text": input_text})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.divider()
                    st.subheader("🔍 Analysis Report")
                    
                    # Verdict Section
                    verdict = data.get('final_verdict', 'Unknown')
                    if "Flagged" in verdict:
                        st.error(f"**Verdict**: {verdict}")
                    else:
                        st.success(f"**Verdict**: {verdict}")
                        
                    col1, col2, col3 = st.columns(3)
                    
                    # 1. Fake News Column
                    with col1:
                        fn = data.get('fake_news_analysis', {})
                        st.info("**Fake News Classifier**")
                        st.metric("Label", fn.get('label', 'N/A'))
                        st.progress(fn.get('confidence', 0))
                        st.caption(f"Confidence: {fn.get('confidence', 0):.2f}")
                        
                    # 2. AI Detection Column
                    with col2:
                        ai = data.get('ai_detection_analysis', {})
                        st.info("**AI Detector**")
                        st.metric("Label", ai.get('label', 'N/A'))
                        st.progress(ai.get('score_ai', 0))
                        st.caption(f"AI Likelihood: {ai.get('score_ai', 0):.2f}")
                        
                    # 3. Consistency Column
                    with col3:
                        cons = data.get('consistency_analysis', {})
                        st.info("**Consistency Check**")
                        score = cons.get('inconsistency_score', 0)
                        st.metric("Inconsistency Score", f"{score:.2f}")
                        if score > 0.3:
                            st.warning("High inconsistency detected.")
                        else:
                            st.success("Consistent text flow.")
                            
                    # Details
                    with st.expander("See Inconsistency Details"):
                        details = cons.get('details', [])
                        if not details:
                            st.write("No significant inconsistencies found.")
                        else:
                            for d in details:
                                st.write(f"**Mismatch between:**")
                                st.text(f"A: {d.get('para_a')}")
                                st.text(f"B: {d.get('para_b')}")
                                st.write(f"*Similarity: {d.get('similarity'):.4f}*")
                                st.divider()
                                
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the Backend API. Make sure `uvicorn app.main:app` is running.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

st.sidebar.markdown("### How it works")
st.sidebar.info(
    "1. **Fake News Model**: RoBERTa fine-tuned on news datasets.\n"
    "2. **AI Detector**: Checks for patterns typical of LLMs.\n"
    "3. **Consistency**: Uses SBERT to find semantic contradictions."
)
