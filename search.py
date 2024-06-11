import os
import arxiv
import requests
import pandas as pd
from scipy import spatial

def arxiv_search(query, embedding_request):
    print(f"Searching arXiv for: {query}")
    try:
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for result in search.results():
            title = result.title
            summary = result.summary
            published = result.published.strftime("%Y-%m-%d")
            url = result.pdf_url
            embedding = embedding_request(summary)
            results.append({
                "title": title,
                "summary": summary,
                "published": published,
                "pdf_url": url,
                "embedding": embedding
            })
        df = pd.DataFrame(results)
        df['relatedness_score'] = df['embedding'].apply(lambda x: 1 - spatial.distance.cosine(embedding_request(query), x))
        df = df.sort_values('relatedness_score', ascending=False)
        df[['title', 'summary', 'published', 'pdf_url', 'relatedness_score']].to_csv(f'arxiv/{query}.csv', index=False, header=False)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []

def google_custom_search(query, embedding_request):
    print(f"Searching CSE for: {query}")
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": os.getenv('GOOGLE_CSE_KEY'),
            "cx": os.getenv('GOOGLE_CSE_ID'),
            "q": query
        }
        response = requests.get(url, params=params)
        data = response.json()
        results = []
        if 'items' in data:
            for item in data['items']:
                title = item['title']
                snippet = item['snippet']
                link = item['link']
                embedding = embedding_request(snippet)
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link,
                    "embedding": embedding
                })
        else:
            print(f"No 'items' found in CSE response. Full response: {data}")
        
        df = pd.DataFrame(results)  
        df['relatedness_score'] = df['embedding'].apply(lambda x: 1 - spatial.distance.cosine(embedding_request(query), x))
        df = df.sort_values('relatedness_score', ascending=False)
        df[['title', 'snippet', 'link', 'relatedness_score']].to_csv(f'cse/{query}.csv', index=False, header=False)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error searching CSE: {e}") 
        return []
