# Investigating-Word-Embedding-Techniques-for-Extracting-Disease-Gene-and-Chemical-Relationships
Overview
This project explores the functional relatedness between biomedical concepts like genes, chemicals, and diseases, focusing on cancer-related research. Using word embeddings generated from a large corpus of PubMed abstracts, we aim to reveal meaningful relationships among these entities. The study evaluates several advanced word embedding models and measures their effectiveness in predicting associations in biomedical data, particularly in cancer research.

Key Objectives
Data Preparation: Process and structure PubMed abstracts to extract relevant biomedical concepts.
Word Embedding Creation: Implement multiple word embedding models, including SkipGram, CBOW, GloVe, PubMedBERT, and BioBERT.
Functional Relatedness Exploration: Analyze the relationships between genes, diseases, and chemicals by calculating cosine similarity.
Evaluation: Measure the performance of each embedding model using precision, recall, and other relevant metrics.
Dataset
Source: PubMed abstracts (1976-2022)
Size: 100,000 cancer-related abstracts
Format: JSON structure, including PubMed IDs, article text, and biomedical annotations (e.g., genes, chemicals, diseases)
Annotation Tool: PubTator (used to identify and annotate biomedical entities)
Workflow
Step 1: Data Preparation
Segment large datasets: Abstracts were split into manageable segments.
Filter data: Focused on extracting relevant biomedical terms (e.g., genes, chemicals, and diseases) using MeSH identifiers.
Create data dictionaries: Organized extracted data into CSV files for further use in word embedding models.
Step 2: Word Embedding Generation
Models Used:

GloVe: Captures global co-occurrence statistics.
Word2Vec (CBOW & SkipGram): Predicts word context using local word co-occurrences.
PubMedBERT: Specialized for biomedical text, leveraging transformer layers.
BioBERT: Focuses on contextual embeddings for biomedical literature.
Process:

Tokenize the text data using NLP tools like nltk.
Match terms from the dataset (genes, chemicals, diseases) to the relevant embedding models.
Generate word embeddings for each biomedical concept.
Store embeddings in structured files organized by model type (e.g., Glove_Disease_combined.csv).
Step 3: Functional Relatedness Evaluation
Cosine Similarity: Compute similarity between gene-disease, gene-chemical, and disease-chemical pairs.
Curated Pairs: Compare embedding-based pairs with manually curated pairs from the Comparative Toxicogenomics Database (CTD).
Thresholds: Apply thresholds (e.g., cosine similarity > 0.6, 0.7, 0.8) to validate associations.
Evaluation Metrics:
Precision: True Positives / (True Positives + False Positives)
Recall: True Positives / (True Positives + False Negatives)
Step 4: Visualization and Comparison
Heatmaps: Visualize precision and recall metrics across different thresholds for each model.
Comparison: Identify the most effective model for detecting functional associations in biomedical data.
Key Findings
Model Performance: PubMedBERT and BioBERT models exhibited superior performance in detecting meaningful relationships between genes, chemicals, and diseases, indicated by higher precision and recall.
Implications: These findings suggest that deep learning models like BERT, fine-tuned on biomedical text, provide valuable insights into functional relatedness in cancer research, aiding in hypothesis generation and drug discovery.
How to Run This Project
Prerequisites
Python 3.7+
Libraries:
pandas
nltk
gensim
transformers
numpy
