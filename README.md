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
a. Segment large datasets: Abstracts were split into manageable segments.
b. Create data dictionaries: Organized extracted data into CSV files for further use in word embedding models to focused on extracting relevant biomedical terms (e.g., genes, chemicals, and diseases) using MeSH identifiers.
Filtering strategy for Disease : C04_588 "Neoplasms by Site" 
Filtering strategy for Chemical : All were selected except Biomedical and Dental Materials" (D25) and "Pharmaceutical Preparations" (D26)
Step 2: Word Embedding Generation
Models Used:
a.GloVe: Captures global co-occurrence statistics.
b.Word2Vec (CBOW & SkipGram): Predicts word context using local word co-occurrences.
c.PubMedBERT: Specialized for biomedical text, leveraging transformer layers.
d. BioBERT: Focuses on contextual embeddings for biomedical literature.
Process:
Tokenize the text data using NLP tools like nltk.
Match terms from the data_sictionaries dataset (genes, chemicals, diseases) to the JSON concept_type field. After matching extracting the term and context around it to create word embeddings.
Generate word embeddings for each biomedical concept.
Store embeddings in structured files organized by model type (e.g., Glove_Disease_combined.csv).
Step 3 : Compare word embeddings saved from different BERT layers:
a. For BERT-based models, word embeddings were saved from different layers, including the summation of the last four hidden layers and the individual embeddings of each of the last four layers. 
b. Cosine similarity was calculated for the embeddings across these layers, and the average similarity between the layers was calculated. 
c. The results were plotted on a bar graph to determine which layers performed best for evaluating functional relatedness. 
Step 4 : Functional Relatedness Evaluation
Cosine Similarity:
a. Compute similarity between gene-disease, gene-chemical, and disease-chemical pairs for curated pairs downloaded from CTD.
b. Compute Cosine Similarity for instance vector pairs created for all instances present from curated CTD.        
c. Thresholds: Apply thresholds (e.g., cosine similarity > 0.6, 0.7, 0.8) to validate associations.
d. Evaluation: Calculate Precision and Recall values for all the models 
Precision: True Positives / (True Positives + False Positives)
Recall: True Positives / (True Positives + False Negatives)
  where, o	True Positives: Instances where the cosine similarity score exceeds the threshold for pairs that are both in the curated CTD data and identified through instance vector pairs, confirming correct identification.
          o	False Negatives: Instances where pairs are recognized in the curated CTD data but either the cosine similarity score does not exceed the threshold or they are not found in instance vector pairs, indicating missed connections.
          o	False Positives: Instances where the cosine similarity score exceeds the threshold for pairs that are not in the curated CTD data suggesting incorrect identifications.
Key Findings:
PubMedBERT embeddings were more similar when compared to BioBERT
e.Visualization and Comparison
Heatmaps: Visualize precision and recall metrics across different thresholds for each model.
Comparison: Identify the most effective model for detecting functional associations in biomedical data.
Key Findings
Model Performance: PubMedBERT and BioBERT models exhibited superior performance in detecting meaningful relationships between genes, chemicals, and diseases, indicated by higher precision and recall.
Implications: These findings suggest that deep learning models like BERT, fine-tuned on biomedical text, provide valuable insights into functional relatedness in cancer research, aiding in hypothesis generation and drug discovery.

Step 5:
Additionally, the study explored whether the word embeddings could capture functional relatedness within the newly introduced 2024 CTD dataset. Only the new pairs present in CTD 2024, but not in previous versions, were filtered. Cosine similarity was then calculated using the previously generated word embeddings.
a.Loading Files: Loading both versions of CTD files referred to as old_ctd and new_ctd 
b. Identifying New Entries: Finding entries present in the new file but not in the old file. 
c. Loading Embeddings: Word embeddings for genes, chemicals, and diseases were loaded individually based on the identified gene-chemical, chemical-disease, and disease-gene pairs. 
d. Cosine Similarity Calculation: For each corresponding pair, the cosine similarity between their embeddings was calculated.

Key Findings:
The results also show that the word embeddings created from PubMed abstracts up to 2022 are able to capture functional relationships in newly curated pairs from the CTD dataset. Specifically, the dataset included 157 disease-chemical pairs, 138 disease-gene pairs, and 191 chemical-gene pairs. Using the generated word embeddings, the model successfully captured relatedness in 42 disease-chemical pairs, 58 disease-gene pairs, and 83 chemical-gene pairs


Prerequisites Required
Libraries:
Python 3.7+
Libraries:
pandas
nltk
gensim
transformers
numpy
Pretrained models: 
a.https://nlp.stanford.edu/projects/glove/
b.https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
c.https://huggingface.co/dmis-lab/biobert-v1.1
Download Dataset from CTD:
You can download curated associations using the batch query: Batch Query | CTD (ctdbase.org). You can use MeSH concept IDs or just disease names for your query. The batch query gives you the option to download curated only, inferred only, or both for all 3 pairs. gene-disease, disease-chemical, chemical-gene. 
Annotated JSON dataset : Pubmed abstaracts annotated properly with fields like PMID, Text, Concept [concept_type, term, identifier] is required. 

Note : In the code provided make sure to check the file names and paths before executing it
The detailed steps will be mentioned in the thesis document. For references. 
