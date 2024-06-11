import requests
import pandas as pd
import json
from scipy import spatial

def google_custom_search(query, embedding_request):
    print(f"Searching CSE for: {query}")
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": "YOUR_API_KEY",
            "cx": "YOUR_SEARCH_ENGINE_ID", 
            "q": query
        }
        data = requests.get(url, params=params).json()
        results = []
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
        df = pd.DataFrame(results)  
        df['relatedness_score'] = df['embedding'].apply(lambda x: 1 - spatial.distance.cosine(embedding_request(query), x))
        df = df.sort_values('relatedness_score', ascending=False)
        df.to_csv(f'cse/{query}.csv', index=False, header=False)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error searching CSE: {e}") 
        return []
