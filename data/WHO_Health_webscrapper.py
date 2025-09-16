import requests
from bs4 import BeautifulSoup
import json

# Top high-priority disease URLs from WHO Fact Sheets
disease_urls = {
    "malaria": "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "dengue": "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    "chikungunya": "https://www.who.int/news-room/fact-sheets/detail/chikungunya",
    "japanese_encephalitis": "https://www.who.int/news-room/fact-sheets/detail/japanese-encephalitis",
    "tuberculosis": "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "polio": "https://www.who.int/news-room/fact-sheets/detail/poliomyelitis",
    "measles": "https://www.who.int/news-room/fact-sheets/detail/measles",
    "covid-19": "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)"
}

kb = {}

for disease, url in disease_urls.items():
    print(f"Scraping {disease}...")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Function to extract text under a heading until next heading
    def extract_section_text(heading_keywords):
        section_text = []
        for header in soup.find_all(['h2', 'h3', 'h4']):
            if any(keyword.lower() in header.get_text(strip=True).lower() for keyword in heading_keywords):
                for sibling in header.find_next_siblings():
                    if sibling.name in ['h2', 'h3', 'h4']:
                        break
                    if sibling.name == 'ul':
                        section_text.extend([li.get_text(strip=True) for li in sibling.find_all('li')])
                    elif sibling.name == 'p':
                        text = sibling.get_text(strip=True)
                        if text:
                            section_text.append(text)
                break
        return section_text

    # Extract symptoms and prevention info
    symptoms = extract_section_text(['symptom', 'signs'])
    prevention = extract_section_text(['prevention', 'control', 'avoid'])

    kb[disease] = {
        "symptoms": symptoms,
        "prevention": prevention
    }

# Save to JSON
with open("health_kb.json", "w", encoding="utf-8") as f:
    json.dump(kb, f, ensure_ascii=False, indent=4)

print("Knowledge Base saved as health_kb.json")
