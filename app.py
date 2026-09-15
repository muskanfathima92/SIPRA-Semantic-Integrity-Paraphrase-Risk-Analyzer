import streamlit as st
import tempfile
import os
import pandas as pd

from src.text_extractor import extract_text
from src.plagiarism import PlagiarismDetector


st.set_page_config(
    page_title="SIPRA",
    page_icon="🛡️",
    layout="wide"
)


# -------------------------
# HEADER
# -------------------------

st.title("🛡️ SIPRA")
st.subheader(
    "Semantic Integrity & Paraphrase Risk Analyzer"
)

st.write(
    "An explainable AI system that detects semantic plagiarism, "
    "paraphrasing and conceptual similarity."
)

st.divider()


# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.header("🧠 AI Analysis")

    st.write("""
    **SIPRA analyzes documents using:**

    ✓ Semantic embeddings  
    ✓ Sentence similarity  
    ✓ Lexical overlap  
    ✓ Paraphrase detection  
    ✓ Explainable evidence  
    ✓ Integrity risk scoring
    """)

    st.divider()

    st.info(
        "Important: Similarity is an analytical signal, "
        "not definitive proof of plagiarism."
    )


# -------------------------
# FILE UPLOAD
# -------------------------

st.header("📂 Compare Documents")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Document A")

    file1 = st.file_uploader(
        "Upload first document",
        type=["pdf", "docx", "txt"],
        key="file1"
    )


with col2:

    st.subheader("Document B")

    file2 = st.file_uploader(
        "Upload second document",
        type=["pdf", "docx", "txt"],
        key="file2"
    )


# -------------------------
# ANALYZE
# -------------------------

if file1 and file2:

    if st.button(
        "🔍 Analyze Semantic Integrity",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing semantic relationships..."
        ):

            try:

                # Temporary files
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(file1.name)[1]
                ) as temp1:

                    temp1.write(file1.getbuffer())
                    path1 = temp1.name


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(file2.name)[1]
                ) as temp2:

                    temp2.write(file2.getbuffer())
                    path2 = temp2.name


                # Extract text
                text1 = extract_text(path1)
                text2 = extract_text(path2)


                # AI detector
                detector = PlagiarismDetector()

                results = detector.analyze(
                    text1,
                    text2
                )


                similarity = results["overall_similarity"]
                matches = results["matches"]
                sentence_results = results["sentences"]


                # -------------------------
                # RISK
                # -------------------------

                if similarity >= 80:
                    risk = "CRITICAL"
                elif similarity >= 60:
                    risk = "HIGH"
                elif similarity >= 30:
                    risk = "MODERATE"
                else:
                    risk = "LOW"


                # -------------------------
                # RESULTS HEADER
                # -------------------------

                st.divider()

                st.header("📊 Semantic Integrity Report")


                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Semantic Overlap",
                    f"{similarity:.1f}%"
                )

                c2.metric(
                    "Matched Sentences",
                    len(matches)
                )

                c3.metric(
                    "Sentences Analyzed",
                    len(sentence_results)
                )

                c4.metric(
                    "Risk Level",
                    risk
                )


                # -------------------------
                # PROGRESS
                # -------------------------

                st.subheader("🧠 Integrity Risk Score")

                st.progress(
                    min(int(similarity), 100)
                )

                if risk == "CRITICAL":

                    st.error(
                        "🚨 Critical semantic overlap detected."
                    )

                elif risk == "HIGH":

                    st.warning(
                        "⚠️ High semantic overlap detected."
                    )

                elif risk == "MODERATE":

                    st.warning(
                        "🟡 Moderate semantic overlap detected."
                    )

                else:

                    st.success(
                        "✅ Low semantic overlap detected."
                    )


                # -------------------------
                # MATCH TYPES
                # -------------------------

                st.divider()

                st.header("🔎 Match Intelligence")

                if matches:

                    match_df = pd.DataFrame(matches)

                    type_counts = (
                        match_df["type"]
                        .value_counts()
                    )

                    cols = st.columns(
                        len(type_counts)
                    )

                    for col, (match_type, count) in zip(
                        cols,
                        type_counts.items()
                    ):

                        col.metric(
                            match_type,
                            count
                        )


                # -------------------------
                # SENTENCE HEATMAP
                # -------------------------

                st.divider()

                st.header(
                    "🌡️ Sentence-Level Originality Heatmap"
                )

                heatmap_data = []

                for i, item in enumerate(
                    sentence_results,
                    start=1
                ):

                    heatmap_data.append({
                        "Sentence": i,
                        "Similarity (%)": item["similarity"],
                        "Classification": item["type"]
                    })

                heatmap_df = pd.DataFrame(
                    heatmap_data
                )

                st.dataframe(
                    heatmap_df,
                    use_container_width=True,
                    hide_index=True
                )


                # -------------------------
                # EVIDENCE
                # -------------------------

                st.divider()

                st.header(
                    "🧾 Explainable Similarity Evidence"
                )

                if not matches:

                    st.success(
                        "No significant semantic matches detected."
                    )

                else:

                    for i, match in enumerate(
                        matches,
                        start=1
                    ):

                        with st.expander(
                            f"Match {i} — "
                            f"{match['type']} — "
                            f"{match['similarity']}%"
                        ):

                            left, right = st.columns(2)

                            with left:

                                st.markdown(
                                    "**Document A**"
                                )

                                st.write(
                                    match["document1"]
                                )

                            with right:

                                st.markdown(
                                    "**Document B**"
                                )

                                st.write(
                                    match["document2"]
                                )

                            st.write(
                                f"🧠 Semantic similarity: "
                                f"**{match['similarity']}%**"
                            )

                            st.write(
                                f"📝 Lexical overlap: "
                                f"**{match['lexical_overlap']}%**"
                            )

                            st.info(
                                "💡 Why flagged: "
                                + match["explanation"]
                            )


                # -------------------------
                # DOCUMENT STATISTICS
                # -------------------------

                st.divider()

                st.header("📄 Document Statistics")

                d1, d2 = st.columns(2)

                with d1:

                    st.metric(
                        "Document A Words",
                        len(text1.split())
                    )

                    st.metric(
                        "Document A Characters",
                        len(text1)
                    )

                with d2:

                    st.metric(
                        "Document B Words",
                        len(text2.split())
                    )

                    st.metric(
                        "Document B Characters",
                        len(text2)
                    )


                # -------------------------
                # DOWNLOAD REPORT
                # -------------------------

                st.divider()

                st.header("📥 Download Analysis")

                report = f"""
SIPRA
Semantic Integrity & Paraphrase Risk Analyzer
==============================================

Document A: {file1.name}
Document B: {file2.name}

Semantic Overlap: {similarity:.2f}%
Risk Level: {risk}

Matched Sentences: {len(matches)}
Sentences Analyzed: {len(sentence_results)}

MATCH EVIDENCE
--------------

"""

                for i, match in enumerate(
                    matches,
                    start=1
                ):

                    report += f"""
Match {i}
Type: {match['type']}
Semantic Similarity: {match['similarity']}%
Lexical Overlap: {match['lexical_overlap']}%

Document A:
{match['document1']}

Document B:
{match['document2']}

Why Flagged:
{match['explanation']}

----------------------------------------
"""


                st.download_button(
                    "📥 Download SIPRA Report",
                    report,
                    file_name="SIPRA_Analysis_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )


                # -------------------------
                # HOW IT WORKS
                # -------------------------

                st.divider()

                st.header("⚙️ How SIPRA Works")

                st.markdown("""
                **1️⃣ Document Extraction**  
                PDF, DOCX and TXT documents are converted into text.

                **2️⃣ Sentence Segmentation**  
                Documents are divided into individual sentences.

                **3️⃣ Semantic Embeddings**  
                Each sentence is converted into a numerical representation using an AI embedding model.

                **4️⃣ Similarity Analysis**  
                Cosine similarity identifies sentences with similar meaning.

                **5️⃣ Paraphrase Intelligence**  
                SIPRA combines semantic similarity with lexical overlap to distinguish direct reuse from paraphrasing.

                **6️⃣ Explainable Evidence**  
                Each flagged match explains why it was identified.

                **7️⃣ Integrity Risk Score**  
                The system converts detected overlap into an easy-to-understand risk indicator.
                """)


            except Exception as e:

                st.error(
                    f"Analysis failed: {e}"
                )

else:

    st.info(
        "👆 Upload two documents to begin semantic integrity analysis."
    )