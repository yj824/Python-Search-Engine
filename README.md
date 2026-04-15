# Python Search Engine

A simple search engine built from scratch in Python for indexing and searching a collection of web pages.

## Overview

This project builds an inverted index from a dataset of web pages and returns ranked search results for user queries. It was developed as an academic information retrieval project and includes indexing, ranking, and duplicate filtering components.

## Features

- Inverted index construction
- Tokenization and Porter stemming
- TF-IDF cosine similarity ranking
- Extra weighting for important HTML elements such as titles, headings, and bold text
- URL-based and title-based relevance boosts
- Exact duplicate filtering
- Near-duplicate filtering using SimHash
- Console-based search interface

## Project Files

- `indexer.py` - builds the inverted index and saves index files
- `search.py` - loads the index and returns ranked search results
- `vocab.pkl` - term-to-ID mapping generated during indexing
- `postings.pkl` - inverted index postings generated during indexing
- `doc_map.pkl` - document ID to URL mapping
- `doc_stats.pkl` - document statistics used for ranking and duplicate filtering

## How It Works

### Indexing
The indexer:
- reads web page data from a dataset directory
- parses HTML content with BeautifulSoup
- tokenizes and stems terms
- applies weighted term frequency based on HTML structure
- stores the processed data as serialized index files

### Searching
The search engine:
- processes the query with tokenization and stemming
- finds candidate documents from the inverted index
- ranks documents using TF-IDF cosine similarity
- applies additional boosts and penalties
- removes exact and near duplicates
- returns the top ranked URLs

## Technologies Used

- Python
- BeautifulSoup
- NLTK
- Pickle
- Regular Expressions

## How to Run

### 1. Install dependencies
```bash
pip install beautifulsoup4 lxml nltk
