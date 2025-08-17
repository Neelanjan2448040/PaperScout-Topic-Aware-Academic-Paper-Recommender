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
        
        # Helper function to format authors for each style
        def format_authors(authors_str: str, style: str) -> str:
            if not authors_str or authors_str == 'Unknown authors':
                return 'Unknown authors'
            
            names = [name.strip() for name in authors_str.split(',')]
            
            if style == "IEEE" or style == "Vancouver":
                formatted_names = []
                for name in names:
                    parts = name.split()
                    last_name = parts[-1]
                    initials = ''.join([part[0].upper() + '.' for part in parts[:-1]])
                    if style == "IEEE":
                        formatted_names.append(f"{initials} {last_name}")
                    else: # Vancouver
                        formatted_names.append(f"{last_name} {initials}")

                if len(formatted_names) > 6:
                    return f"{formatted_names[0]} et al."
                else:
                    return ', '.join(formatted_names)
            
            elif style == "Harvard":
                formatted_names = []
                for name in names:
                    parts = name.split()
                    last_name = parts[-1]
                    initials = '. '.join([part[0].upper() for part in parts[:-1]]) + '.' if len(parts) > 1 else ''
                    formatted_names.append(f"{last_name}, {initials}".strip())
                
                if len(formatted_names) > 3:
                    return f"{formatted_names[0].split(',')[0]} et al."
                else:
                    return ', '.join(formatted_names)

            # Default to APA/MLA author format
            return authors_str

        formatted_authors = format_authors(authors, style)

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
        
        elif style == "IEEE":
            citation = f"{formatted_authors}, “{title},”"
            if url:
                citation += f" [Online]. Available: {url}."
            if doi:
                citation += f" doi: {doi}."
            if not url and not doi:
                 citation += f" ({year})." # Simple format for papers without DOI or URL
            return citation
        
        elif style == "Vancouver":
            citation = f"{formatted_authors}. {title}. "
            if year:
                citation += f"{year}. "
            if url:
                citation += f"Available from: {url}. "
            if doi:
                citation += f"doi:{doi}."
            return citation.strip()
        
        elif style == "Harvard":
            citation = f"{formatted_authors} ({year}) '{title}'."
            if url:
                citation += f" Available at: {url}."
            if doi:
                citation += f" doi: {doi}."
            return citation

        else:
            return f"{authors} ({year}). {title}"

# The rest of your app.py code would follow here...