# app.py
# 🚀 Force PyTorch mode (no TensorFlow/Keras for this script)
import os
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import math
import sqlite3
import datetime as dt
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer

# Embeddings + ANN
try:
    import faiss # faiss-cpu
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ---------- CONFIG ----------
APP_NAME = "PaperScout+: Topic-Aware Academic Paper Recommender"
DB_PATH = "paperscout_cache.sqlite"
USER_AGENT = "PaperScoutPlus/1.0 (mailto:your_email@example.com)"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# INCREASED: Fetch more papers per API call to reach the 200+ target
RESULTS_PER_SOURCE = 100 # Increased to get a larger pool of papers
MAX_RESULTS_SHOW = 500 # Set max range to 500
REQUEST_TIMEOUT = 30 # Increased timeout for larger API calls

SEMANTIC_SCHOLAR_FIELDS = "title,abstract,year,authors,url,externalIds,citationCount,publicationTypes,publicationVenue,fieldsOfStudy,references"

# New list of common publication types
PUBLICATION_TYPES = ["Article", "Review", "Book", "Book Chapter", "Conference Paper", "Dataset", "Preprint", "Patent", "Thesis", "Any"]

# Expanded list of available sources for more coverage
SOURCES = ["semanticscholar", "arxiv", "crossref", "europepmc", "pubmed", "core", "doaj", "openalex"]

# Expanded list of domains
DOMAINS = ["Any", "AI", "NLP", "Computer Vision", "Machine Learning", "Data Science", "Physics", "Mathematics", "Biology", "Medicine", "Chemistry", "Economics", "Social Science", "Signal Processing", "Robotics", "Quantum Computing", "Genetics", "Neuroscience"]

# ---------- CITATION MANAGER ----------
class CitationManager:
    """Simple citation manager class"""
    
    @staticmethod
    def generate_citation(paper: Dict[str, Any], style: str) -> str:
        """Generate citation in specified style"""
        title = paper.get('title', 'No title')
        authors = paper.get('authors', 'Unknown authors')
        year = paper.get('year', 'n.d.')
        doi = paper.get('doi', '')
        url = paper.get('url', '')
        
        if style == "APA":
            citation = f"{authors} ({year}). {title}."
            if doi:
                citation += f" https://doi.org/{doi}"
            elif url:
                citation += f" Retrieved from {url}"
            return citation
        elif style == "MLA":
            citation = f'{authors}. "{title}." {year}.'
            if url:
                citation += f" Web. {url}"
            return citation
        elif style == "Chicago":
            citation = f'{authors}. "{title}." Accessed {year}.'
            if url:
                citation += f" {url}."
            return citation
        elif style == "BibTeX":
            # Clean up the title and authors for BibTeX
            clean_title = title.replace('{', '').replace('}', '')
            clean_authors = authors.replace('{', '').replace('}', '')
            
            # Generate a citation key
            first_author = clean_authors.split(',')[0].split()[-1] if clean_authors else "Unknown"
            cite_key = f"{first_author}{year}".replace(' ', '')
            
            bibtex = f"""@article{{{cite_key},
    title = {{{clean_title}}},
    author = {{{clean_authors}}},
    year = {{{year}}},"""
            
            if doi:
                bibtex += f"\n    doi = {{{doi}}},"
            if url:
                bibtex += f"\n    url = {{{url}}},"
                
            bibtex += "\n}"
            return bibtex
            
        else:
            return f"{authors} ({year}). {title}"

# ---------- UTILITIES ----------
def init_db():
    """Initializes or connects to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            key TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            authors TEXT,
            year INTEGER,
            doi TEXT,
            url TEXT,
            source TEXT,
            categories TEXT,
            publication_type TEXT,
            citation_count INTEGER,
            added_at TEXT,
            references_ids TEXT
        )
    """)
    
    # Safer way to add columns
    try:
        cur.execute("ALTER TABLE papers ADD COLUMN publication_type TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cur.execute("ALTER TABLE papers ADD COLUMN references_ids TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            q TEXT,
            params TEXT,
            results TEXT,
            created_at TEXT,
            PRIMARY KEY (q, params)
        )
    """)
    conn.commit()
    return conn

def norm_text(x: Optional[str]) -> str:
    """Normalizes a string, returning an empty string if None."""
    if x is None:
        return ""
    return str(x).strip()

def make_key(doi: Optional[str], title: str, year: Optional[int]) -> str:
    """Creates a unique key for a paper, prioritizing DOI."""
    if doi and doi.strip():
        return doi.lower().strip()
    base = (title or "").lower().strip()
    y = str(year or "")
    return f"h::{hash(base + '|' + y)}"

def safe_int(x, default=None):
    """Safely converts a value to an integer."""
    if x is None:
        return default
    try:
        return int(float(x))  # Handle string numbers
    except (ValueError, TypeError):
        return default

def current_iso():
    """Returns the current UTC time in ISO format."""
    return dt.datetime.utcnow().isoformat()

def clamp01(x):
    """Clamps a value between 0 and 1."""
    try:
        return max(0.0, min(1.0, float(x)))
    except (ValueError, TypeError):
        return 0.0

def hybrid_score(similarity, year, citations, min_year, max_year):
    """
    Calculates a hybrid score blending semantic similarity, recency, and citation count.
    Fixed version with better error handling.
    """
    # Ensure all inputs are valid
    similarity = clamp01(similarity or 0)
    year = safe_int(year)
    citations = safe_int(citations, 0) or 0
    min_year = safe_int(min_year, 2000)
    max_year = safe_int(max_year, dt.datetime.utcnow().year)
    
    year_score = 0.0
    current_year = dt.datetime.utcnow().year
    
    if year is not None and max_year > min_year:  # Fixed: prevent division by zero
        recency = (year - min_year) / (max_year - min_year)
        if year >= current_year - 2:
            recency *= 1.2  # Slight boost for recent papers
        year_score = clamp01(recency)
    elif year is not None and year >= current_year - 2:
        year_score = 0.8  # High score for recent papers when all papers are from same year
    
    citation_score = 0.0
    if citations > 0:
        citation_score = clamp01(math.log10(citations + 1) / math.log10(1000 + 1))  # Adjusted scale
    
    # Dynamic weighting
    s_weight = 0.6
    y_weight = 0.2
    c_weight = 0.2
    
    # Adjust weights based on data availability
    if citations == 0 and max_year == min_year:
        s_weight = 0.8
        y_weight = 0.2
        c_weight = 0.0
    elif citations == 0:
        s_weight = 0.7
        y_weight = 0.3
        c_weight = 0.0
    elif max_year == min_year:
        s_weight = 0.7
        y_weight = 0.0
        c_weight = 0.3
    
    return (similarity * s_weight) + (year_score * y_weight) + (citation_score * c_weight)

# ---------- API FETCHERS ----------
def fetch_arxiv(query: str, max_results: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Fixed arXiv fetcher with better error handling"""
    import xml.etree.ElementTree as ET
    
    try:
        # Better query formatting for arXiv
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        start = 0
        size = min(max_results, 100)
        
        url = f"http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:"{clean_query}"',
            'start': start,
            'max_results': size
        }
        
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, 
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        
        root = ET.fromstring(r.text)
        ns = {'a': 'http://www.w3.org/2005/Atom'}
        
        entries = []
        for entry in root.findall('a:entry', ns):
            try:
                title = norm_text(entry.findtext('a:title', default="", namespaces=ns))
                if not title:
                    continue
                    
                summary = norm_text(entry.findtext('a:summary', default="", namespaces=ns))
                
                # Find the correct link
                link = ""
                for l in entry.findall('a:link', ns):
                    if l.attrib.get('rel') == 'alternate':
                        link = l.attrib.get('href', '')
                        break
                
                # Parse publication date
                published = entry.findtext('a:published', default="", namespaces=ns)
                year = None
                if published:
                    try:
                        year = int(published[:4])
                    except (ValueError, IndexError):
                        pass
                
                # Parse authors
                authors = []
                for a in entry.findall('a:author', ns):
                    name = norm_text(a.findtext('a:name', default="", namespaces=ns))
                    if name:
                        authors.append(name)
                
                # Parse categories
                cats = []
                for c in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                    term = c.attrib.get('term', '')
                    if term:
                        cats.append(term)
                
                # Look for DOI
                doi = ""
                for doi_elem in entry.findall('{http://arxiv.org/schemas/atom}doi'):
                    if doi_elem is not None and doi_elem.text:
                        doi = norm_text(doi_elem.text)
                        break
                
                entries.append({
                    "title": title,
                    "abstract": summary,
                    "authors": ", ".join(authors),
                    "year": year,
                    "doi": doi,
                    "url": link,
                    "source": "arxiv",
                    "categories": ", ".join(cats),
                    "publication_type": "Preprint",
                    "citation_count": 0,  # arXiv doesn't provide citation counts
                    "references_ids": "[]"
                })
            except Exception as e:
                # Skip problematic entries
                continue
                
        return entries
    except Exception as e:
        print(f"arXiv fetch error: {str(e)}")
        return []

def fetch_crossref(query: str, rows: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Improved CrossRef fetcher"""
    try:
        url = "https://api.crossref.org/works"
        params = {
            'query': query,
            'rows': min(rows, 1000),  # CrossRef limit
            'select': 'title,abstract,author,issued,DOI,URL,type,reference'
        }
        
        headers = {
            "User-Agent": USER_AGENT,
            "mailto": "your_email@example.com"
        }
        
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers=headers)
        r.raise_for_status()
        
        data = r.json().get("message", {}).get("items", [])
        out = []
        
        for it in data:
            try:
                title = norm_text(" ".join(it.get("title", []) or []))
                if not title:
                    continue
                
                abstract = norm_text(it.get("abstract") or "")
                # Clean HTML tags from abstract
                abstract = re.sub(r"<[^<]+?>", "", abstract)
                
                # Parse authors
                authors = []
                for a in (it.get("author", []) or []):
                    given = a.get("given", "")
                    family = a.get("family", "")
                    if given and family:
                        authors.append(f"{given} {family}")
                    elif family:
                        authors.append(family)
                
                # Parse year
                year = None
                issued = it.get("issued", {})
                if "date-parts" in issued and issued["date-parts"]:
                    try:
                        year = safe_int(issued["date-parts"][0][0])
                    except (IndexError, TypeError):
                        pass
                
                doi = norm_text(it.get("DOI") or "")
                url_field = norm_text(it.get("URL") or "")
                pub_type = norm_text(it.get("type", "").replace('-', ' ').title())
                
                # Parse references
                references = []
                for ref in (it.get('reference', []) or []):
                    if ref.get('DOI'):
                        references.append(ref['DOI'])
                
                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors),
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "crossref",
                    "categories": pub_type,
                    "publication_type": pub_type or "Article",
                    "citation_count": 0,  # CrossRef doesn't provide citation counts in this endpoint
                    "references_ids": json.dumps(references)
                })
            except Exception as e:
                continue
                
        return out
    except Exception as e:
        print(f"CrossRef fetch error: {str(e)}")
        return []

def fetch_europe_pmc(query: str, page_size: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Improved Europe PMC fetcher"""
    try:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            'query': query,
            'format': 'json',
            'pageSize': min(page_size, 1000),
            'resultType': 'core'
        }
        
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, 
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        
        data = r.json().get("resultList", {}).get("result", []) or []
        out = []
        
        for it in data:
            try:
                title = norm_text(it.get("title") or "")
                if not title:
                    continue
                
                abstract = norm_text(it.get("abstractText") or "")
                authors = norm_text(it.get("authorString") or "")
                year = safe_int(it.get("pubYear"))
                doi = norm_text(it.get("doi") or "")
                
                # Build URL
                pmid = it.get("pmid", "")
                url_field = f"https://europepmc.org/abstract/MED/{pmid}" if pmid else ""
                
                categories = norm_text(it.get("journalTitle") or it.get("pubType") or "")
                pub_type = norm_text(it.get("pubType") or "Article")
                
                # Parse references
                references = []
                for ref in (it.get("references", []) or []):
                    ref_id = ref.get("id")
                    if ref_id:
                        references.append(ref_id)
                
                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "europepmc",
                    "categories": categories,
                    "publication_type": pub_type,
                    "citation_count": 0,
                    "references_ids": json.dumps(references)
                })
            except Exception as e:
                continue
                
        return out
    except Exception as e:
        print(f"Europe PMC fetch error: {str(e)}")
        return []

def fetch_semantic_scholar(query: str, limit: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Improved Semantic Scholar fetcher"""
    try:
        cleaned_query = re.sub(r'[\r\n\t]+', ' ', query).strip()
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        params = {
            'query': cleaned_query,
            'limit': min(limit, 80),  # API limit
            'fields': SEMANTIC_SCHOLAR_FIELDS
        }
        
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, 
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        
        data = r.json().get("data", []) or []
        out = []
        
        for it in data:
            try:
                title = norm_text(it.get("title") or "")
                if not title:
                    continue
                
                abstract = norm_text(it.get("abstract") or "")
                year = safe_int(it.get("year"))
                url_field = norm_text(it.get("url") or "")
                citation_count = safe_int(it.get("citationCount"), 0) or 0
                
                # Parse fields of study
                fields = ", ".join(it.get("fieldsOfStudy") or [])
                
                # Parse authors
                authors = []
                for a in (it.get("authors") or []):
                    name = a.get("name", "")
                    if name:
                        authors.append(name)
                
                # Parse DOI
                doi = ""
                ext = it.get("externalIds") or {}
                if "DOI" in ext and ext["DOI"]:
                    doi = str(ext["DOI"])
                
                # Parse publication types
                pub_types = it.get("publicationTypes") or []
                pub_type = ", ".join(pub_types) if pub_types else "Article"
                
                # Parse references
                references = []
                for ref in (it.get("references", []) or []):
                    paper_id = ref.get("paperId")
                    if paper_id:
                        references.append(paper_id)
                
                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors),
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "semanticscholar",
                    "categories": fields,
                    "publication_type": pub_type,
                    "citation_count": citation_count,
                    "references_ids": json.dumps(references)
                })
            except Exception as e:
                continue
                
        return out
    except Exception as e:
        # Silently fail, do not show warning to user
        print(f"Semantic Scholar fetch error: {str(e)}")
        return []

def fetch_pubmed(query: str, max_results: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Improved PubMed fetcher with better error handling"""
    try:
        import xml.etree.ElementTree as ET
        
        # Search for paper IDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 200),  # PubMed limit
            'retmode': 'json'
        }
        
        r = requests.get(esearch_url, params=params, timeout=REQUEST_TIMEOUT, 
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        
        if not ids:
            return []
        
        # Fetch details for found papers
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(ids[:50]),  # Limit to avoid timeout
            'retmode': 'xml'
        }
        
        r = requests.get(efetch_url, params=params, timeout=REQUEST_TIMEOUT,
                         headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        
        root = ET.fromstring(r.text)
        out = []
        
        for article in root.findall('.//PubmedArticle'):
            try:
                medline_citation = article.find('MedlineCitation')
                if medline_citation is None:
                    continue
                    
                article_info = medline_citation.find('Article')
                if article_info is None:
                    continue
                
                # Extract title
                title_elem = article_info.find('ArticleTitle')
                title = norm_text(title_elem.text if title_elem is not None else "")
                if not title:
                    continue
                
                # Extract abstract
                abstract = ""
                abs_element = article_info.find('Abstract/AbstractText')
                if abs_element is not None and abs_element.text:
                    abstract = norm_text(abs_element.text)
                
                # Extract year
                year = None
                year_elem = article_info.find('Journal/JournalIssue/PubDate/Year')
                if year_elem is not None and year_elem.text:
                    year = safe_int(year_elem.text)
                
                # Extract authors
                authors_list = []
                for author_elem in article_info.findall('AuthorList/Author'):
                    last_name_elem = author_elem.find('LastName')
                    first_name_elem = author_elem.find('ForeName')
                    if last_name_elem is not None and last_name_elem.text:
                        name = last_name_elem.text
                        if first_name_elem is not None and first_name_elem.text:
                            name = f"{first_name_elem.text} {name}"
                        authors_list.append(name)
                
                # Extract DOI
                doi = ""
                for id_elem in medline_citation.findall('.//ELocationID'):
                    if id_elem.get('EIdType') == 'doi' and id_elem.text:
                        doi = norm_text(id_elem.text)
                        break
                
                # Build URL
                pmid_elem = medline_citation.find('PMID')
                pmid = pmid_elem.text if pmid is not None else ""
                url_field = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                
                # Extract publication type
                pub_type = "Article"
                pub_type_elem = article_info.find('PublicationTypeList/PublicationType')
                if pub_type_elem is not None and pub_type_elem.text:
                    pub_type = norm_text(pub_type_elem.text)
                
                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(authors_list),
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "pubmed",
                    "categories": "Medicine",
                    "publication_type": pub_type,
                    "citation_count": 0,  # PubMed doesn't provide citation counts
                    "references_ids": "[]"
                })
            except Exception as e:
                continue
                
        return out
    except Exception as e:
        print(f"PubMed fetch error: {str(e)}")
        return []

def fetch_core(query: str, limit: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Fetches papers from the CORE API."""
    try:
        url = "https://api.core.ac.uk/v3/search/works"
        headers = {
            "User-Agent": USER_AGENT,
            # Add a CORE API key if you have one for a higher rate limit
            # "Authorization": "Bearer YOUR_CORE_API_KEY"
        }
        
        # The CORE API uses a POST request for search
        payload = {
            "q": query,
            "limit": min(limit, 100)
        }
        
        r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        
        data = r.json().get("results", [])
        out = []
        
        for it in data:
            try:
                work = it.get("work", {})
                title = norm_text(work.get("title") or "")
                if not title:
                    continue

                abstract = norm_text(work.get("abstract") or "")
                authors = ", ".join([a.get("name", "") for a in work.get("authors", [])])
                year = safe_int(work.get("year"))
                doi = norm_text(work.get("doi") or "")
                url_field = norm_text(work.get("url") or "")
                
                # CORE provides citation data
                citation_count = safe_int(work.get("citationCount", 0))

                pub_type = norm_text(work.get("publicationType") or "Article")
                
                # Categories from CORE
                categories = ", ".join([c.get("displayName", "") for c in work.get("topics", [])])

                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "core",
                    "categories": categories,
                    "publication_type": pub_type,
                    "citation_count": citation_count,
                    "references_ids": "[]"
                })
            except Exception as e:
                continue
                
        return out
    except Exception as e:
        # Silently fail, do not show warning to user
        print(f"CORE fetch error: {str(e)}")
        return []

def fetch_doaj(query: str, page_size: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Fetches articles from the DOAJ API."""
    try:
        url = "https://doaj.org/api/v2/articles/search"
        headers = {
            "User-Agent": USER_AGENT,
        }
        
        params = {
            "query": query,
            "pageSize": min(page_size, 100)
        }
        
        r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        
        data = r.json().get("results", [])
        out = []
        
        for it in data:
            try:
                # DOAJ structure is nested
                bibjson = it.get("bibjson", {})
                
                title = norm_text(bibjson.get("title") or "")
                if not title:
                    continue

                abstract = norm_text(bibjson.get("abstract") or "")
                authors = ", ".join([a.get("name", "") for a in bibjson.get("author", [])])
                
                year = None
                date_str = bibjson.get("journal", {}).get("publication_start_date") or bibjson.get("year")
                if date_str:
                    try:
                        year = safe_int(date_str[:4])
                    except:
                        pass
                
                doi = bibjson.get("identifier", [{}])[0].get("id") if bibjson.get("identifier") else ""
                url_field = bibjson.get("link", [{}])[0].get("url") if bibjson.get("link") else ""
                
                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "doaj",
                    "categories": ", ".join(bibjson.get("subject", []) or []),
                    "publication_type": "Journal Article",
                    "citation_count": 0, # DOAJ does not provide citation counts
                    "references_ids": "[]"
                })
            except Exception as e:
                continue
                
        return out
    except Exception as e:
        # Silently fail, do not show warning to user
        print(f"DOAJ fetch error: {str(e)}")
        return []

def fetch_openalex(query: str, limit: int = RESULTS_PER_SOURCE) -> List[Dict[str, Any]]:
    """Fetches papers from the OpenAlex API."""
    try:
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": min(limit, 200),
            "select": "id,title,authorships,publication_year,doi,cited_by_count,primary_location,concepts,type"
        }
        
        # Add email for polite pool
        headers = {
            "User-Agent": USER_AGENT,
            "mailto": "your_email@example.com"
        }

        r = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        data = r.json().get("results", [])
        out = []

        for it in data:
            try:
                title = norm_text(it.get("title") or "")
                if not title:
                    continue
                
                # Fetch abstract separately as it's not in the default select
                # Note: This increases API calls and may be slow. A better approach for a production app is to include it in the initial request if possible, or use a separate API call with OR syntax.
                abstract = ""
                # abstract_url = f"https://api.openalex.org/works/{it['id']}"
                # abs_r = requests.get(abstract_url, headers=headers, timeout=REQUEST_TIMEOUT)
                # if abs_r.status_code == 200:
                #    abstract = norm_text(abs_r.json().get("abstract", "text") or "")
                
                authors = ", ".join([a.get("author", {}).get("display_name", "") for a in it.get("authorships", [])])
                year = safe_int(it.get("publication_year"))
                
                doi = norm_text(it.get("doi") or "")
                url_field = it.get("primary_location", {}).get("source", {}).get("full_text_url") or it.get("primary_location", {}).get("landing_page_url") or ""
                
                citation_count = safe_int(it.get("cited_by_count", 0))

                concepts = ", ".join([c.get("display_name", "") for c in it.get("concepts", [])])
                pub_type = norm_text(it.get("type", "").replace('-', ' ').title())

                out.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "doi": doi,
                    "url": url_field,
                    "source": "openalex",
                    "categories": concepts,
                    "publication_type": pub_type,
                    "citation_count": citation_count,
                    "references_ids": "[]" # References would require a separate, slower API call
                })
            except Exception as e:
                continue
        
        return out
    except Exception as e:
        print(f"OpenAlex fetch error: {str(e)}")
        return []

# ---------- CACHE LAYER ----------
def upsert_papers(conn, papers: List[Dict[str, Any]]):
    """Inserts or updates papers in the local database cache."""
    if not papers:
        return
        
    cur = conn.cursor()
    for p in papers:
        try:
            title = norm_text(p.get("title"))
            if not title:  # Skip papers without titles
                continue
                
            year = safe_int(p.get("year"))
            doi = norm_text(p.get("doi"))
            key = make_key(doi, title, year)
            
            cur.execute("""
                INSERT OR REPLACE INTO papers 
                (key, title, abstract, authors, year, doi, url, source, categories, publication_type, citation_count, added_at, references_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                title,
                norm_text(p.get("abstract")),
                norm_text(p.get("authors")),
                year,
                doi,
                norm_text(p.get("url")),
                norm_text(p.get("source")),
                norm_text(p.get("categories")),
                norm_text(p.get("publication_type")),
                safe_int(p.get("citation_count"), 0) or 0,
                current_iso(),
                norm_text(p.get("references_ids", "[]"))
            ))
        except Exception as e:
            continue  # Skip problematic papers
            
    conn.commit()

def load_cached_paper_by_key(conn, key: str) -> Optional[pd.Series]:
    """Loads a single cached paper by its key."""
    try:
        cur = conn.cursor()
        cur.execute("""SELECT key,title,abstract,authors,year,doi,url,source,categories,publication_type,citation_count,references_ids 
                        FROM papers WHERE key=?""", (key,))
        row = cur.fetchone()
        if row:
            cols = ["key","title","abstract","authors","year","doi","url","source","categories","publication_type","citation_count", "references_ids"]
            return pd.Series(dict(zip(cols, row)))
        return None
    except Exception:
        return None

def find_papers_by_id(conn, ids: List[str]) -> pd.DataFrame:
    """Finds papers in the cache by their IDs (DOIs or other unique keys)."""
    if not ids:
        return pd.DataFrame()
    
    try:
        # Create a list of '?' placeholders for the query
        placeholders = ','.join(['?' for _ in ids])
        query = f"SELECT * FROM papers WHERE doi IN ({placeholders}) OR key IN ({placeholders})"
        
        cur = conn.cursor()
        results = cur.execute(query, ids + ids).fetchall()
        
        if not results:
            return pd.DataFrame()
        
        cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(results, columns=cols)
    except Exception:
        return pd.DataFrame()

def query_cached(conn, q: str, params: Dict[str,Any]) -> Optional[List[Dict[str,Any]]]:
    """Retrieves cached search results for a query and its parameters."""
    try:
        cur = conn.cursor()
        key_params = json.dumps(params, sort_keys=True)
        cur.execute("SELECT results, created_at FROM query_cache WHERE q=? AND params=?", (q, key_params))
        r = cur.fetchone()
        if r:
            try:
                return json.loads(r[0])
            except Exception:
                return None
        return None
    except Exception:
        return None

def save_query_cache(conn, q: str, params: Dict[str,Any], results: List[Dict[str,Any]]):
    """Saves search results to the query cache."""
    try:
        cur = conn.cursor()
        key_params = json.dumps(params, sort_keys=True)
        cur.execute("INSERT OR REPLACE INTO query_cache (q, params, results, created_at) VALUES (?, ?, ?, ?)", 
                   (q, key_params, json.dumps(results), current_iso()))
        conn.commit()
    except Exception:
        pass  # Fail silently for cache operations

# ---------- MERGE & DEDUP ----------
def merge_dedup(papers: List[Dict[str,Any]]) -> pd.DataFrame:
    """Merges and deduplicates a list of papers from different sources."""
    if not papers:
        return pd.DataFrame()
        
    try:
        df = pd.DataFrame(papers)
        if df.empty:
            return df
        
        # Ensure all required columns exist
        required_cols = ["title","abstract","authors","doi","url","source","categories","publication_type", "references_ids"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""
        
        # Clean and normalize text columns
        for col in required_cols:
            df[col] = df[col].fillna("").astype(str).str.strip()
        
        # Handle numeric columns
        df["year"] = df["year"].apply(safe_int)
        df["citation_count"] = df["citation_count"].apply(lambda x: safe_int(x, 0) or 0)
        
        # Create keys for deduplication
        def compute_key(row):
            return make_key(row["doi"], row["title"], row["year"])
        
        df["key"] = df.apply(compute_key, axis=1)
        
        # Remove obvious duplicates
        df = df.drop_duplicates(subset=["key"])
        
        # Additional deduplication by title and year
        df = df.sort_values(by=["doi", "citation_count"], ascending=[True, False])
        df = df.drop_duplicates(subset=["title","year"], keep="first")
        
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error in merge_dedup: {str(e)}")
        return pd.DataFrame()

# ---------- EMBEDDINGS + SEARCH ----------
@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the SentenceTransformer model and caches it."""
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def build_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Encodes a list of texts into a numpy array of embeddings."""
    if not texts or model is None:
        return np.zeros((0, 384), dtype=np.float32)
    
    try:
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return np.zeros((len(texts), 384), dtype=np.float32)
        
        emb = model.encode(valid_texts, batch_size=32, show_progress_bar=False, 
                           normalize_embeddings=True, convert_to_numpy=True)
        
        # Handle case where some texts were empty
        if len(valid_texts) < len(texts):
            full_emb = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
            valid_idx = 0
            for i, text in enumerate(texts):
                if text.strip():
                    full_emb[i] = emb[valid_idx]
                    valid_idx += 1
            return full_emb
        
        return emb.astype("float32")
    except Exception as e:
        st.error(f"Error building embeddings: {str(e)}")
        return np.zeros((len(texts), 384), dtype=np.float32)

def build_index(emb: np.ndarray):
    """Builds a FAISS or NumPy index for a given set of embeddings."""
    if emb.shape[0] == 0:
        return None
    
    try:
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            return index
        else:
            return emb
    except Exception as e:
        st.warning(f"Error building search index: {str(e)}")
        return emb  # Fallback to numpy

def search_index(index, query_vec: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a similarity search on the index."""
    if index is None or query_vec.size == 0:
        return np.array([]), np.array([])
    
    try:
        if FAISS_AVAILABLE and hasattr(index, 'search'):
            scores, idxs = index.search(query_vec.reshape(1, -1), top_k)
            return scores[0], idxs[0]
        else:
            # Numpy fallback
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(-1, 1)
            scores = (index @ query_vec).reshape(-1)
            idxs = np.argsort(-scores)[:top_k]
            return scores[idxs], idxs
    except Exception as e:
        st.warning(f"Error in similarity search: {str(e)}")
        return np.array([]), np.array([])

# ---------- MAIN FETCH + RECOMMEND ----------
def unified_search(query: str, year_range: Tuple[int,int], domain_filter: str, 
                   pub_types: List[str], sources: List[str], min_citations: int, 
                   max_results: int) -> pd.DataFrame:
    """Performs a unified search across multiple sources with filtering and ranking."""
    processed_query = query.strip()
    if not processed_query:
        return pd.DataFrame()

    try:
        conn = init_db()
        params = {
            "year_range": year_range, 
            "domain": domain_filter, 
            "pub_types": sorted(pub_types), 
            "sources": sorted(sources), 
            "min_citations": min_citations, 
            "max_results": max_results
        }
        
        # Check cache first
        cached = query_cached(conn, processed_query, params)
        if cached is not None:
            return pd.DataFrame(cached)
        
        # Fetch from selected sources
        results: List[Dict[str,Any]] = []
        
        source_fetchers = {
            "semanticscholar": fetch_semantic_scholar,
            "arxiv": fetch_arxiv,
            "crossref": fetch_crossref,
            "europepmc": fetch_europe_pmc,
            "pubmed": fetch_pubmed,
            "core": fetch_core,
            "doaj": fetch_doaj,
            "openalex": fetch_openalex
        }
        
        with st.spinner("Searching academic databases..."):
            for source in sources:
                if source in source_fetchers:
                    try:
                        results += source_fetchers[source](processed_query, RESULTS_PER_SOURCE)
                    except Exception as e:
                        # Silently handle errors
                        print(f"Error fetching from {source.upper()}: {e}")

        # Merge and deduplicate
        df = merge_dedup(results)
        if df.empty:
            save_query_cache(conn, processed_query, params, [])
            return df

        # Cache raw results
        upsert_papers(conn, df.to_dict(orient="records"))

        # Apply filters
        if min_citations > 0:
            df = df[df["citation_count"] >= int(min_citations)]
        
        if domain_filter and domain_filter != "Any":
            mask = df["categories"].str.contains(domain_filter, case=False, na=False, regex=False)
            df = df[mask]

        if pub_types and "Any" not in pub_types:
            pattern = "|".join([re.escape(pt) for pt in pub_types])
            mask = df["publication_type"].str.contains(pattern, case=False, na=False, regex=True)
            df = df[mask]
        
        # Year filter
        year_mask = (df["year"].fillna(0) >= year_range[0]) & (df["year"].fillna(9999) <= year_range[1])
        df = df[year_mask]

        if df.empty:
            save_query_cache(conn, processed_query, params, [])
            return df

        # Semantic search and ranking
        model = load_model()
        if model is None:
            return df.head(max_results)
        
        with st.spinner("Ranking papers by relevance..."):
            texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
            emb = build_embeddings(texts, model)
            
            if emb.shape[0] == 0:
                save_query_cache(conn, processed_query, params, [])
                return df.head(max_results)
            
            index = build_index(emb)
            qvec = build_embeddings([processed_query], model)
            
            if qvec.shape[0] == 0:
                return df.head(max_results)
            
            sims, idxs = search_index(index, qvec[0], top_k=min(max_results, len(df)))

            if len(idxs) == 0:
                save_query_cache(conn, processed_query, params, [])
                return df.head(max_results)
            
            # Apply ranking
            df2 = df.iloc[idxs].copy().reset_index(drop=True)
            
            # Calculate year range for scoring
            year_values = df2["year"].dropna()
            year_min = int(year_values.min()) if len(year_values) > 0 else 2000
            year_max = int(year_values.max()) if len(year_values) > 0 else dt.datetime.utcnow().year
            
            df2["similarity"] = sims[:len(df2)]
            df2["hybrid_score"] = df2.apply(
                lambda r: hybrid_score(
                    float(r["similarity"]), 
                    safe_int(r["year"]), 
                    safe_int(r["citation_count"], 0), 
                    year_min, 
                    year_max
                ), axis=1
            )
            
            df2 = df2.sort_values("hybrid_score", ascending=False).head(max_results).reset_index(drop=True)
            
            # Cache final results
            save_query_cache(conn, processed_query, params, df2.to_dict(orient="records"))
            return df2
            
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return pd.DataFrame()

def get_similar_papers_by_content(parent_paper: pd.Series, top_k: int = 10) -> pd.DataFrame:
    """Finds papers similar to a parent paper by performing a new semantic search."""
    if parent_paper is None or len(parent_paper) == 0:
        st.warning("No paper data provided for similar paper search.")
        return pd.DataFrame()
    
    query = str(parent_paper.get("title", "")) + " " + str(parent_paper.get("abstract", ""))
    query = query.strip()
    if not query:
        st.warning("No title or abstract found for this paper to perform a similar paper search.")
        return pd.DataFrame()

    try:
        with st.spinner(f"Finding papers similar to '{parent_paper.get('title', 'this paper')}'..."):
            all_results = []
            
            # Fetch from multiple sources
            fetchers = [fetch_semantic_scholar, fetch_arxiv, fetch_crossref, fetch_core, fetch_doaj, fetch_openalex]
            for fetch_func in fetchers:
                try:
                    all_results += fetch_func(query, RESULTS_PER_SOURCE // len(fetchers))
                except Exception:
                    pass

            if not all_results:
                st.warning("No similar papers found.")
                return pd.DataFrame()

            df = merge_dedup(all_results)
            if df.empty:
                return pd.DataFrame()
            
            # Cache results
            upsert_papers(init_db(), df.to_dict(orient="records"))

            # Perform semantic search
            model = load_model()
            if model is None:
                return df.head(top_k)

            texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()
            emb = build_embeddings(texts, model)
            
            if emb.shape[0] == 0:
                return pd.DataFrame()
            
            index = build_index(emb)
            qvec = build_embeddings([query], model)
            
            if qvec.shape[0] == 0:
                return df.head(top_k)
            
            sims, idxs = search_index(index, qvec[0], top_k=min(top_k, len(df)))

            if len(idxs) == 0:
                return pd.DataFrame()

            df2 = df.iloc[idxs].copy().reset_index(drop=True)
            
            # Calculate scores
            year_values = df2["year"].dropna()
            year_min = int(year_values.min()) if len(year_values) > 0 else 2000
            year_max = int(year_values.max()) if len(year_values) > 0 else dt.datetime.utcnow().year

            df2["similarity"] = sims[:len(df2)]
            df2["hybrid_score"] = df2.apply(
                lambda r: hybrid_score(
                    float(r["similarity"]), 
                    safe_int(r["year"]), 
                    safe_int(r["citation_count"], 0), 
                    year_min, 
                    year_max
                ), axis=1
            )

            return df2.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
            
    except Exception as e:
        st.error(f"Error finding similar papers: {str(e)}")
        return pd.DataFrame()

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title=APP_NAME, layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #4f46e5;
    --primary-light: #6366f1;
    --secondary: #10b981;
    --background: #f8fafc;
    --card-bg: #ffffff;
    --text-dark: #1f2937;
    --text-gray: #6b7280;
    --border: #e5e7eb;
    --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Base App */
.stApp {
    font-family: 'Inter', sans-serif;
    background: var(--background);
    color: var(--text-dark);
}

/* Main Container */
.main .block-container {
    padding: 2rem 1rem;
    max-width: 1200px;
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: var(--shadow);
    margin: 1rem auto;
}

/* Headers */
h1 {
    color: var(--primary) !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 1rem !important;
    font-size: 2.5rem !important;
}

h2, h3 {
    color: var(--text-dark) !important;
    font-weight: 600 !important;
}

/* Sidebar */
.css-1d391kg {
    background: var(--card-bg) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow) !important;
    padding: 1.5rem !important;
    margin: 1rem !important;
}

/* Paper Cards - Clean & Readable */
.paper-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: all 0.3s ease;
    position: relative;
}

.paper-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-light);
}

.paper-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--primary-light));
    border-radius: 12px 12px 0 0;
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.paper-card:hover::before {
    transform: scaleX(1);
}

/* Paper Title Links */
.paper-title a {
    color: var(--primary) !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    line-height: 1.4 !important;
    display: block !important;
}

.paper-title a:hover {
    color: var(--primary-light) !important;
    text-decoration: underline !important;
}

/* Paper Meta Info */
.paper-meta {
    color: var(--text-gray) !important;
    font-size: 0.9rem !important;
    margin: 0.5rem 0 !important;
    font-weight: 500 !important;
}

/* Paper Abstract */
.paper-abstract {
    color: var(--text-dark) !important;
    line-height: 1.6 !important;
    font-size: 0.95rem !important;
    margin-top: 1rem !important;
}

/* Buttons - Clean & Interactive */
.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'Inter' !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background: var(--primary-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Input Fields - Clear & Focused */
.stTextInput > div > div > input {
    border: 2px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
    transition: border-color 0.2s ease !important;
    background: var(--card-bg) !important;
    color: var(--text-dark) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
}

/* Select Boxes */
.stSelectbox > div > div > select {
    border: 2px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    background: var(--card-bg) !important;
    color: var(--text-dark) !important;
}

/* Multiselect */
.stMultiSelect > div > div {
    border: 2px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--card-bg) !important;
}

/* Progress Bars */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--secondary), var(--primary)) !important;
    border-radius: 10px !important;
}

/* Metrics Cards */
div[data-testid="metric-container"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    box-shadow: var(--shadow) !important;
    transition: transform 0.2s ease !important;
}

div[data-testid="metric-container"]:hover {
    transform: scale(1.02) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--background) !important;
    padding: 4px !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: var(--text-gray) !important;
    font-weight: 500 !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* Expanders - Clear Content */
.streamlit-expanderHeader {
    background: var(--background) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
    font-weight: 600 !important;
}

.streamlit-expanderContent {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    color: var(--text-dark) !important;
    padding: 1rem !important;
}

/* Info Boxes */
.stInfo {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
}

.stSuccess {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
}

.stWarning {
    background: #fffbeb !important;
    border: 1px solid #fed7aa !important;
    border-radius: 8px !important;
    color: var(--text-dark) !important;
}

/* Data Editor */
.stDataFrame {
    border-radius: 8px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow) !important;
}

.stDataFrame table {
    background: var(--card-bg) !important;
    color: var(--text-dark) !important;
}

.stDataFrame th {
    background: var(--background) !important;
    color: var(--text-dark) !important;
    font-weight: 600 !important;
}

/* Charts */
.js-plotly-plot {
    background: var(--card-bg) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
}

/* Sidebar Labels */
.stSidebar label {
    color: var(--text-dark) !important;
    font-weight: 600 !important;
}

/* Download Button Special */
.stDownloadButton > button {
    background: var(--secondary) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: #059669 !important;
    transform: translateY(-1px) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--background);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--text-gray);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* Responsive */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem 0.5rem;
        margin: 0.5rem;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    .paper-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
}

/* Ensure all text is readable */
* {
    color: var(--text-dark) !important;
}

p, span, div {
    color: var(--text-dark) !important;
}

/* Override any invisible text */
.stMarkdown, .stText {
    color: var(--text-dark) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 PaperScout+: Topic-Aware Academic Paper Recommender")
st.caption("Real-time multi-source search • Semantic ranking • Trend insight")

# Initialize session state variables
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()
if "similar_papers_df" not in st.session_state:
    st.session_state.similar_papers_df = pd.DataFrame()
if "similar_paper_parent" not in st.session_state:
    st.session_state.similar_paper_parent = None
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "year_range" not in st.session_state:
    st.session_state.year_range = (2015, dt.datetime.utcnow().year)
if "domain" not in st.session_state:
    st.session_state.domain = "Any"
if "citations_range" not in st.session_state:
    st.session_state.citations_range = (0, 5000)
if "pub_types" not in st.session_state:
    st.session_state.pub_types = ["Article", "Review", "Conference Paper", "Preprint"]
if "sources" not in st.session_state:
    st.session_state.sources = SOURCES.copy()
if "max_results" not in st.session_state:
    st.session_state.max_results = MAX_RESULTS_SHOW
if "select_all" not in st.session_state:
    st.session_state.select_all = False

def run_search_with_query():
    """Triggers the search with the current query."""
    st.session_state.current_query = st.session_state.query_input
    if st.session_state.current_query and st.session_state.sources:
        st.session_state.results_df = unified_search(
            query=st.session_state.current_query.strip(),
            year_range=st.session_state.year_range,
            domain_filter=st.session_state.domain,
            pub_types=st.session_state.pub_types,
            sources=st.session_state.sources,
            min_citations=st.session_state.citations_range[0],
            max_results=st.session_state.max_results
        )
        st.session_state.filtered_df = st.session_state.results_df.copy()
        st.session_state.similar_papers_df = pd.DataFrame()
        st.session_state.similar_paper_parent = None
        st.session_state.select_all = False # Reset select all on new search
    elif not st.session_state.sources:
        st.warning("Please select at least one source to search.")
        st.session_state.results_df = pd.DataFrame()
        st.session_state.filtered_df = pd.DataFrame()
    else:
        st.warning("Please enter a query to start a search.")
        st.session_state.results_df = pd.DataFrame()
        st.session_state.filtered_df = pd.DataFrame()

with st.sidebar:
    st.header("🔎 Search Filters")
    st.markdown("---")
    
    st.text_area(
        "Enter your research topic, keywords, or abstract.", 
        placeholder="e.g., Explainable AI for medical imaging and diagnosis",
        key="query_input"
    )
    
    go = st.button("Search", type="primary", on_click=run_search_with_query)

    st.markdown("---")
    st.subheader("Advanced Filters")
    current_year = dt.datetime.utcnow().year
    st.slider("Year range", min_value=1990, max_value=current_year, 
              value=st.session_state.year_range, step=1, key="year_range")
    st.selectbox("Domain/category (contains)", DOMAINS, key="domain")
    st.markdown("### Citation Range")
    st.slider("Citations range", min_value=0, max_value=5000, 
              value=st.session_state.citations_range, step=10, key="citations_range")
    st.multiselect(
        "Publication Type", 
        options=PUBLICATION_TYPES, 
        default=["Article", "Review", "Conference Paper", "Preprint"],
        key="pub_types"
    )
    selected_sources = st.multiselect(
        "Search Sources", 
        options=SOURCES, 
        default=SOURCES,
        key="sources"
    )
    st.slider("Results to show (max 500)", 50, 500, MAX_RESULTS_SHOW, 10, key="max_results")

# Display similar papers if a similar papers search was triggered
if not st.session_state.similar_papers_df.empty:
    parent_title = "Unknown Title"
    if st.session_state.similar_paper_parent is not None:
        parent_title = st.session_state.similar_paper_parent.get('title', 'Unknown Title')
    
    st.subheader(f"Similar Papers to: {parent_title}")
    
    for i, row in st.session_state.similar_papers_df.iterrows():
        with st.container():
            st.markdown(f'<div class="paper-card">', unsafe_allow_html=True)
            title = row["title"] or "(No title)"
            url = row["url"] or ""
            if url:
                st.markdown(f'### <span class="paper-title">{i+1}. <a href="{url}" target="_blank">{title}</a></span>', unsafe_allow_html=True)
            else:
                st.markdown(f'### {i+1}. {title}')
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown(f'<div class="paper-meta">✍️ **Authors:** {row.get("authors","N/A") or "N/A"}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="paper-meta">📅 **Year:** {row.get("year","N/A") or "N/A"}</div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="paper-meta">📈 **Citations:** {row.get("citation_count", 0) or 0}</div>', unsafe_allow_html=True)
            
            with st.expander("📖 **Show Abstract**"):
                abstract = row["abstract"] or "No abstract available."
                st.markdown(f'<div class="paper-abstract">{abstract}</div>', unsafe_allow_html=True)
            
            st.markdown(f"**Source:** {row.get('source','').upper()} &nbsp; • &nbsp; **Type:** {row.get('publication_type','N/A') or 'N/A'}")
            st.markdown(f"**Categories:** *{row['categories'] or 'N/A'}*")
            
            if "hybrid_score" in row:
                score = float(row.get("hybrid_score", 0.0))
                st.progress(score)
                sim_score = float(row.get("similarity", 0.0))
                st.markdown(f"**Final Score:** `{score:.2f}` (Similarity: `{sim_score:.2f}`)")

            st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("⬅️ Back to Main Search Results"):
        st.session_state.similar_papers_df = pd.DataFrame()
        st.session_state.similar_paper_parent = None
        st.rerun()

# Display main results
elif not st.session_state.results_df.empty:
    st.subheader("Search Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Results", len(st.session_state.results_df))
    
    year_values = st.session_state.results_df["year"].dropna()
    if len(year_values) > 0:
        yr_min = int(year_values.min())
        yr_max = int(year_values.max())
        c2.metric("Year Span", f"{yr_min} – {yr_max}")
    else:
        c2.metric("Year Span", "N/A")
    
    citation_values = st.session_state.results_df["citation_count"].dropna()
    if len(citation_values) > 0:
        avg_citations = int(citation_values.mean())
        c3.metric("Avg Citations", f"{avg_citations:,}")
    else:
        c3.metric("Avg Citations", "N/A")
        
    c4.metric("Sources", len(st.session_state.results_df["source"].unique()))
    
    st.divider()

    tabs = st.tabs(["🔎 Recommendations", "📈 Trends", "⬇️ Download"])

    with tabs[0]:
        st.subheader("Top Recommended Papers 🏆")
        st.info("Results are ranked by a **hybrid score** that blends **semantic similarity**, **recency**, and **citation count**.")
        
        filter_col, sort_col, search_col = st.columns([1, 1, 2])
        
        with search_col:
            reco_search_query = st.text_input(
                "Search in recommended papers:", 
                placeholder="Search by title, author, or abstract...",
                key="reco_search"
            )

        with sort_col:
            sort_by = st.selectbox(
                "Sort by", 
                options=["Hybrid Score", "Year", "Citation Count", "Title"],
                key="reco_sort_by"
            )

        temp_df = st.session_state.results_df.copy()
        if reco_search_query:
            search_mask = temp_df.apply(
                lambda row: reco_search_query.lower() in (
                    str(row["title"]) + str(row["authors"]) + str(row["abstract"])
                ).lower(), axis=1
            )
            temp_df = temp_df[search_mask]
        
        if sort_by == "Hybrid Score":
            temp_df = temp_df.sort_values("hybrid_score", ascending=False)
        elif sort_by == "Year":
            temp_df = temp_df.sort_values("year", ascending=False, na_position='last')
        elif sort_by == "Citation Count":
            temp_df = temp_df.sort_values("citation_count", ascending=False)
        elif sort_by == "Title":
            temp_df = temp_df.sort_values("title", ascending=True)

        st.session_state.filtered_df = temp_df

        if st.session_state.filtered_df.empty:
            st.warning("No papers match the current search or filter criteria.")
        else:
            for i, (idx, row) in enumerate(st.session_state.filtered_df.iterrows()):
                with st.container():
                    st.markdown(f'<div class="paper-card">', unsafe_allow_html=True)
                    title = row["title"] or "(No title)"
                    url = row["url"] or ""
                    if url:
                        st.markdown(f'### <span class="paper-title">{i+1}. <a href="{url}" target="_blank">{title}</a></span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'### {i+1}. {title}')
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.markdown(f'<div class="paper-meta">✍️ **Authors:** {row.get("authors","N/A") or "N/A"}</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="paper-meta">📅 **Year:** {row.get("year","N/A") or "N/A"}</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="paper-meta">📈 **Citations:** {row.get("citation_count", 0) or 0}</div>', unsafe_allow_html=True)
                    
                    with st.expander("📖 **Show Abstract**"):
                        abstract = row["abstract"] or "No abstract available."
                        st.markdown(f'<div class="paper-abstract">{abstract}</div>', unsafe_allow_html=True)
                    
                    st.markdown(f"**Source:** {row.get('source','').upper()} &nbsp; • &nbsp; **Type:** {row.get('publication_type','N/A') or 'N/A'}")
                    st.markdown(f"**Categories:** *{row['categories'] or 'N/A'}*")
                    
                    if "hybrid_score" in row:
                        score = float(row.get("hybrid_score", 0.0))
                        st.progress(score)
                        sim_score = float(row.get("similarity", 0.0))
                        st.markdown(f"**Final Score:** `{score:.2f}` (Similarity: `{sim_score:.2f}`)")

                    st.markdown("---")
                    if st.button(f"🔍 Find Similar Papers", key=f"similar_btn_{idx}_{i}"):
                        st.session_state.similar_paper_parent = row
                        st.session_state.similar_papers_df = get_similar_papers_by_content(row)
                        st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.subheader("Trends & Analytics")
        df_tr = st.session_state.results_df.copy()
        df_tr = df_tr.dropna(subset=["year"])
        if df_tr.empty:
            st.info("No year information available for trend analysis.")
        else:
            try:
                st.markdown("### Publication Count by Year")
                counts = df_tr.groupby("year").size().reset_index(name="count")
                fig = px.bar(counts, x="year", y="count", 
                           title="Papers per Year in Results", 
                           color_discrete_sequence=["#007bff"])
                fig.update_layout(xaxis_title="Year", yaxis_title="Number of Papers")
                st.plotly_chart(fig, use_container_width=True)
                
                # Only show citation trends if we have citation data
                citation_data = df_tr[df_tr["citation_count"] > 0]
                if not citation_data.empty:
                    st.markdown("### Citations by Year (Median)")
                    citations = citation_data.groupby("year")["citation_count"].median().reset_index(name="median_citations")
                    fig_citations = px.line(citations, x="year", y="median_citations", 
                                             title="Median Citations by Year", 
                                             markers=True, color_discrete_sequence=["#28a745"])
                    fig_citations.update_layout(xaxis_title="Year", yaxis_title="Median Citations")
                    st.plotly_chart(fig_citations, use_container_width=True)
                
                st.markdown("### Paper Sources Breakdown")
                source_counts = df_tr.groupby("source").size().reset_index(name="count")
                fig_sources = px.pie(source_counts, values="count", names="source", 
                                     title="Distribution of Papers by Source")
                st.plotly_chart(fig_sources, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating trends: {str(e)}")

    with tabs[2]:
        st.subheader("Export Papers")
        st.markdown("Select papers to download as an Excel file with citation details.")
        
        if st.session_state.results_df.empty:
            st.warning("Please run a search first to get results to export.")
        else:
            st.markdown("---")
            
            # Create filter controls above the data editor
            filter_cols = st.columns(3)
            
            with filter_cols[0]:
                download_search_query = st.text_input(
                    "Search table:", 
                    placeholder="Search titles or authors...",
                    key="download_search_query"
                )
            
            with filter_cols[1]:
                year_values = st.session_state.results_df['year'].dropna()
                if len(year_values) > 0:
                    min_year = int(year_values.min())
                    max_year = int(year_values.max())
                else:
                    min_year = 1990
                    max_year = dt.datetime.utcnow().year
                    
                download_year_range = st.slider(
                    "Filter by Year:", 
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    step=1,
                    key='download_year_filter'
                )

            with filter_cols[2]:
                available_types = [t for t in st.session_state.results_df['publication_type'].unique() 
                                 if pd.notnull(t) and str(t).strip()]
                if not available_types:
                    available_types = ["Article"]
                    
                download_pub_type = st.multiselect(
                    "Filter by Type:",
                    options=available_types,
                    default=available_types,
                    key='download_pub_type'
                )

            # Apply filtering
            filtered_download_df = st.session_state.results_df.copy()
            
            if download_search_query:
                search_mask = filtered_download_df.apply(
                    lambda row: download_search_query.lower() in (
                        str(row['title']) + str(row['authors'])
                    ).lower(), axis=1
                )
                filtered_download_df = filtered_download_df[search_mask]
            
            # Year filtering
            year_mask = ((filtered_download_df['year'].fillna(0) >= download_year_range[0]) & 
                         (filtered_download_df['year'].fillna(9999) <= download_year_range[1]))
            filtered_download_df = filtered_download_df[year_mask]
            
            # Publication type filtering
            if download_pub_type:
                type_mask = filtered_download_df['publication_type'].isin(download_pub_type)
                filtered_download_df = filtered_download_df[type_mask]

            if filtered_download_df.empty:
                st.warning("No papers match the current filters.")
            else:
                # 🚀 FIX: Reset the index of the filtered DataFrame to prevent IndexingError
                papers_for_editor = filtered_download_df.copy().reset_index(drop=True)
                papers_for_editor['Select'] = False
                
                # Add a 'Select All' checkbox, managing state with session_state
                select_all = st.checkbox("Select All Visible Papers", value=st.session_state.select_all, key="select_all_checkbox")
                st.session_state.select_all = select_all
                
                if st.session_state.select_all:
                    papers_for_editor['Select'] = True

                # Prepare columns for data editor
                display_cols = ['Select', 'title', 'authors', 'year', 'citation_count']
                editor_df = papers_for_editor[display_cols].copy()
                
                # Fill NaN values for display
                editor_df = editor_df.fillna({'title': 'No title', 'authors': 'Unknown', 
                                              'year': 0, 'citation_count': 0})

                edited_df = st.data_editor(
                    editor_df,
                    hide_index=True,
                    column_config={
                        "Select": st.column_config.CheckboxColumn(
                            "Select",
                            help="Select papers to export",
                            default=False,
                        ),
                        "title": "Title",
                        "authors": "Authors", 
                        "year": "Year",
                        "citation_count": "Citations"
                    },
                    disabled=['title', 'authors', 'year', 'citation_count'],
                    key="download_data_editor"
                )
                
                # 🚀 FIX: Use the updated DataFrame with the aligned index
                selected_papers_df = filtered_download_df.iloc[edited_df[edited_df['Select']].index].copy()

                
                st.markdown("---")
                
                if not selected_papers_df.empty:
                    st.success(f"Selected {len(selected_papers_df)} papers for export.")
                    
                    citation_style = st.selectbox(
                        "Choose Citation Style:",
                        options=["APA", "MLA", "Chicago", "BibTeX", "IEEE", "Vancouver", "Harvard"],
                        key="citation_style_select"
                    )
                    
                    if st.button("Generate & Download Excel File"):
                        try:
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                export_data = selected_papers_df[["title", "abstract", "authors", "year", "doi", "url"]].copy()
                                
                                # Generate citations
                                export_data["citation"] = selected_papers_df.apply(
                                    lambda row: CitationManager.generate_citation(row.to_dict(), citation_style),
                                    axis=1
                                )
                                
                                # Clean up the data for export
                                final_export = export_data.fillna("")
                                final_export.to_excel(writer, index=False, sheet_name='Selected Papers')
                            
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="⬇️ Download Selected Papers as Excel",
                                data=excel_data,
                                file_name=f"paperscout_export_{citation_style.lower()}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Error generating Excel file: {str(e)}")
                else:
                    st.info("No papers selected for export.")
            
            st.markdown("---")
            st.markdown("You can also download **all** papers in BibTeX format.")
            
            def convert_to_bibtex(df: pd.DataFrame) -> str:
                """Convert DataFrame to BibTeX format"""
                bibtex_string = ""
                for _, row in df.iterrows():
                    try:
                        bibtex_string += CitationManager.generate_citation(row.to_dict(), "BibTeX")
                    except Exception:
                        continue  # Skip problematic entries
                return bibtex_string
                
            if not st.session_state.results_df.empty:
                try:
                    bibtex_output = convert_to_bibtex(st.session_state.results_df)
                    st.download_button(
                        "Download All as BibTeX", 
                        data=bibtex_output.encode('utf-8'), 
                        file_name="paperscout_results.bib", 
                        mime="application/x-bibtex"
                    )
                except Exception as e:
                    st.error(f"Error generating BibTeX: {str(e)}")

# Show help message when no results
elif st.session_state.current_query:
    st.info("No results found. Try adjusting your search terms or filters.")
else:
    st.markdown("""
    ### Welcome to PaperScout+! 👋
    
    Get started by entering your research topic in the sidebar and clicking **Search**.
    
    **Features:**
    - 🔍 **Multi-source search** across major academic databases
    - 🎯 **Semantic ranking** using AI to find most relevant papers  
    - 📊 **Trend analysis** to visualize publication patterns
    - 📥 **Export options** with formatted citations
    - 🔗 **Similar paper discovery** for deeper exploration
    
    **Tips for better results:**
    - Use specific keywords related to your research area
    - Combine multiple terms (e.g., "machine learning medical diagnosis")
    - Adjust filters to narrow down results by year, citation count, etc.
    - Try different sources if one doesn't return good results
    """)

st.markdown("---")
st.caption("Sources: Semantic Scholar, CrossRef, arXiv, Europe PMC, PubMed, CORE, DOAJ, OpenAlex · Embeddings: all-MiniLM-L6-v2 · ANN: FAISS")

# Add error boundary and cleanup
try:
    # Cleanup any temporary resources
    pass
except Exception as e:
    st.error(f"Cleanup error: {str(e)}")