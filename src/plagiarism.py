import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlagiarismDetector:

    def __init__(self):
        print("Starting SIPRA Hybrid Analysis Engine...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
        print("SIPRA engine ready!")

    # -----------------------------
    # SENTENCE SPLITTING
    # -----------------------------

    def split_sentences(self, text):

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        return [
            s.strip()
            for s in sentences
            if len(s.strip()) > 10
        ]

    # -----------------------------
    # WORD OVERLAP
    # -----------------------------

    def lexical_overlap(self, a, b):

        words1 = set(
            re.findall(r'\b\w+\b', a.lower())
        )

        words2 = set(
            re.findall(r'\b\w+\b', b.lower())
        )

        if not words1 or not words2:
            return 0

        return (
            len(words1 & words2)
            / len(words1 | words2)
        ) * 100

    # -----------------------------
    # TEXT STRUCTURE SIMILARITY
    # -----------------------------

    def structural_similarity(self, a, b):

        return (
            SequenceMatcher(
                None,
                a.lower(),
                b.lower()
            ).ratio()
            * 100
        )

    # -----------------------------
    # CLASSIFICATION
    # -----------------------------

    def classify_match(
        self,
        hybrid_score,
        lexical_score
    ):

        if (
            hybrid_score >= 75
            and lexical_score >= 50
        ):
            return "Direct Reuse"

        elif hybrid_score >= 45:

            return "Possible Paraphrase"

        elif hybrid_score >= 30:

            return "Conceptual Similarity"

        return "Low Similarity"

    # -----------------------------
    # EXPLANATION
    # -----------------------------

    def explain_match(
        self,
        hybrid_score,
        lexical_score,
        match_type
    ):

        if match_type == "Direct Reuse":

            return (
                "High similarity with substantial "
                "word overlap suggests possible "
                "direct reuse."
            )

        if match_type == "Possible Paraphrase":

            return (
                "The sentences contain similar "
                "information and structure while "
                "using different wording. This "
                "pattern may indicate paraphrasing."
            )

        if match_type == "Conceptual Similarity":

            return (
                "The sentences appear to discuss "
                "related concepts, but the evidence "
                "is weaker than direct reuse."
            )

        return (
            "Only limited similarity was detected."
        )

    # -----------------------------
    # MAIN ANALYSIS
    # -----------------------------

    def analyze(self, text1, text2):

        sentences1 = self.split_sentences(text1)
        sentences2 = self.split_sentences(text2)

        if not sentences1 or not sentences2:

            return {
                "overall_similarity": 0,
                "matches": [],
                "sentences": []
            }

        # TF-IDF representation
        all_sentences = (
            sentences1 + sentences2
        )

        vectors = self.vectorizer.fit_transform(
            all_sentences
        )

        vectors1 = vectors[
            :len(sentences1)
        ]

        vectors2 = vectors[
            len(sentences1):
        ]

        tfidf_matrix = cosine_similarity(
            vectors1,
            vectors2
        )

        matches = []
        sentence_results = []

        for i, sentence1 in enumerate(
            sentences1
        ):

            best_index = (
                tfidf_matrix[i].argmax()
            )

            tfidf_score = (
                tfidf_matrix[i][best_index]
                * 100
            )

            sentence2 = sentences2[
                best_index
            ]

            # Word overlap
            lexical_score = (
                self.lexical_overlap(
                    sentence1,
                    sentence2
                )
            )

            # Structural similarity
            structure_score = (
                self.structural_similarity(
                    sentence1,
                    sentence2
                )
            )

            # Hybrid score
            hybrid_score = (
                (tfidf_score * 0.40)
                +
                (structure_score * 0.60)
            )

            match_type = (
                self.classify_match(
                    hybrid_score,
                    lexical_score
                )
            )

            explanation = (
                self.explain_match(
                    hybrid_score,
                    lexical_score,
                    match_type
                )
            )

            sentence_results.append({

                "sentence": sentence1,

                "similarity": round(
                    hybrid_score,
                    2
                ),

                "type": match_type

            })

            if hybrid_score >= 30:

                matches.append({

                    "document1": sentence1,

                    "document2": sentence2,

                    "similarity": round(
                        hybrid_score,
                        2
                    ),

                    "lexical_overlap": round(
                        lexical_score,
                        2
                    ),

                    "type": match_type,

                    "explanation": explanation

                })

        if matches:

            overall = (
                sum(
                    x["similarity"]
                    for x in matches
                )
                / len(matches)
            )

        else:

            overall = 0

        return {

            "overall_similarity": round(
                overall,
                2
            ),

            "matches": matches,

            "sentences": sentence_results

        }