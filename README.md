# RAG Clusters: Analyzing Perplexity's Source Citation Patterns

## Overview

This project explores the probabilistic nature of Large Language Models (LLMs) by analyzing how Perplexity AI selects and cites sources when generating responses. Using unsupervised learning techniques, we investigate whether distinct clusters of sources emerge based on citation frequency patterns.

## Research Question

Does the probabilistic nature of LLMs extend to the sources they use to ground their answers? While we know LLM responses vary with each query, this project examines whether source selection also demonstrates consistent patterns or remains truly random.

## Dataset

- **Size**: ~1,000 responses from Perplexity AI
- **Topic**: Master of Science in Data Science programs (online and in-person)
- **Query Design**: 20 different aspects (reviews, comparisons, trends, etc.) to ensure comprehensive topic coverage
- **Data Files**:
  - `cu_queries_responses.csv`: Contains queries, aspects, and Perplexity responses
  - `cu_citations.csv`: Contains cited URLs, domains, and response IDs

## Methodology

### Unsupervised Learning Approach
- **Primary Algorithm**: K-means clustering with bootstrap validation
- **Distance Metric**: Citation frequency counts
- **Validation**: Bootstrap resampling methodology adapted from Nugent & Stuetzle

### Key Analysis Steps
1. **Data Collection**: 1,000 Perplexity API responses across 20 query aspects
2. **Exploratory Data Analysis**: Citation patterns, frequency distributions, domain analysis
3. **Clustering**: Bootstrap k-means to identify source citation clusters
4. **Validation**: Statistical confidence intervals for cluster stability

## Key Findings

- **Citation Distribution**: Small number of frequently cited sources vs. many infrequently cited sources
- **Early Appearance**: Most frequently cited domains appear within the first 10-20 responses
- **Probabilistic Behavior**: New sources continue to emerge even after 1,000 responses
- **Domain vs. URL Patterns**: Different citation patterns at domain level vs. specific URL level

## Project Structure

```
├── citation_clusters.ipynb     # Main analysis notebook
├── bootstrapped_clustering.py  # Bootstrap k-means implementation
├── data/
│   ├── cu_citations.csv       # Citation data
│   └── cu_queries_responses.csv # Query and response data
├── images/                    # Screenshots and plots
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Environment Setup
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `citation_clusters.ipynb`
3. Run all cells to reproduce the analysis

## Dependencies

- **pandas** (2.2.3): Data manipulation and analysis
- **numpy** (2.0.2): Numerical computing
- **scipy** (1.14.1): Statistical functions
- **matplotlib** (3.9.2): Data visualization
- **scikit-learn** (1.5.2): Machine learning algorithms
- **kneed** (0.8.5): Knee/elbow detection for optimal k selection
- **ipykernel** (7.1.0): Jupyter notebook support

## Results & Insights

The analysis reveals that Perplexity's source selection follows predictable patterns despite the probabilistic nature of LLMs:

1. **Clustering Behavior**: Sources can be meaningfully clustered into high, medium, and low frequency groups
2. **Early Dominance**: High-frequency sources establish dominance early in the response sequence
3. **Long Tail**: Continuous emergence of new sources demonstrates ongoing probabilistic exploration

## Technical Implementation

### Bootstrap K-means
- Implements confidence interval-based clustering validation
- Uses resampling to assess cluster stability
- Provides statistical confidence in cluster assignments

### Knee Detection
- Automated selection of optimal k using elbow method
- Identifies natural breakpoints in citation frequency distributions

## Future Work

- Expand analysis to other AI answer engines (Claude, ChatGPT, etc.)
- Investigate topic-specific citation patterns
- Explore temporal changes in source selection over time
- Compare citation patterns across different query types

## Academic Context

This project was completed as part of the **Unsupervised Algorithms in Machine Learning** course in the **Master of Science in Data Science** program.

## License

This project is for academic purposes. Data collection follows Perplexity AI's terms of service.

## Contact

For questions about this analysis or the methodology, please refer to the detailed implementation in `citation_clusters.ipynb`.