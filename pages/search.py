import arxiv
import pandas as pd
import json
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
        df.to_csv(f'arxiv/{query}.csv', index=False, header=False)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []
