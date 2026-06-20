import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

def clean_html_text(html_content):
    if pd.isna(html_content): return ""
    try:
        # Extract text from HTML
        soup = BeautifulSoup(str(html_content), 'html.parser')
        text = soup.get_text(separator=' ')
        # Keep only alphabetic words
        words = re.findall(r'[a-zA-Z]{3,}', text)
        return " ".join(words).lower()
    except:
        return ""

def clean_url(url):
    if pd.isna(url): return ""
    # Extract words from URL (ignore www, http, https, com, etc. using stopwords later)
    words = re.findall(r'[a-zA-Z]{3,}', str(url))
    return " ".join(words).lower()

def generate_wordcloud(text_series, title, filename, colormap='viridis'):
    print(f"Generating {title}...")
    text = " ".join(text_series.astype(str).tolist())
    
    custom_stopwords = set(STOPWORDS).union({
        'http', 'https', 'www', 'com', 'org', 'net', 'html', 'php', 'login', 'index', 'htm'
    })
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          stopwords=custom_stopwords,
                          colormap=colormap,
                          max_words=150).generate(text)
                          
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=0)
    plt.savefig(filename)
    plt.close()

def main():
    out_dir = "docs/assets/wordclouds"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading datasets...")
    # Loading Old Data (Use a sample of 2000 for speed)
    df_old_url = pd.read_excel('data/raw/URL.xlsx').head(2000)
    df_old_html = pd.read_excel('data/raw/html.xlsx').head(2000)
    if 'Label' in df_old_url.columns:
        df_old_url = df_old_url.rename(columns={'Label': 'Category'})
        
    df_old = pd.DataFrame({
        'url': df_old_url['Data'],
        'html': df_old_html['Data'],
        'category': df_old_url['Category']
    })
    
    # Loading New Data
    df_new_url = pd.read_excel('data/raw/OOD_URL.xlsx')
    df_new_html = pd.read_excel('data/raw/OOD_html.xlsx')
    
    df_new = pd.DataFrame({
        'url': df_new_url['Data'],
        'html': df_new_html['Data'],
        'category': df_new_url['Category']
    })
    
    print("Cleaning URL texts...")
    df_old['url_clean'] = df_old['url'].apply(clean_url)
    df_new['url_clean'] = df_new['url'].apply(clean_url)
    
    print("Cleaning HTML texts (this takes a moment)...")
    # Sample HTML to save time on BeautifulSoup parsing
    df_old['html_clean'] = [clean_html_text(x) for x in tqdm(df_old['html'])]
    df_new['html_clean'] = [clean_html_text(x) for x in tqdm(df_new['html'])]
    
    # 1. Old URLs (Legit vs Phishing)
    generate_wordcloud(df_old[df_old['category'] == 'ham']['url_clean'], 
                       "2021 Legitimate URLs", os.path.join(out_dir, "old_legit_url.png"), colormap='Greens')
    generate_wordcloud(df_old[df_old['category'] == 'spam']['url_clean'], 
                       "2021 Phishing URLs", os.path.join(out_dir, "old_phish_url.png"), colormap='Reds')
                       
    # 2. New URLs (Legit vs Phishing)
    generate_wordcloud(df_new[df_new['category'] == 'ham']['url_clean'], 
                       "2026 Legitimate URLs", os.path.join(out_dir, "new_legit_url.png"), colormap='Greens')
    generate_wordcloud(df_new[df_new['category'] == 'spam']['url_clean'], 
                       "2026 Phishing URLs", os.path.join(out_dir, "new_phish_url.png"), colormap='Reds')

    # 3. Old HTMLs (Legit vs Phishing)
    generate_wordcloud(df_old[df_old['category'] == 'ham']['html_clean'], 
                       "2021 Legitimate HTML Text", os.path.join(out_dir, "old_legit_html.png"), colormap='Blues')
    generate_wordcloud(df_old[df_old['category'] == 'spam']['html_clean'], 
                       "2021 Phishing HTML Text", os.path.join(out_dir, "old_phish_html.png"), colormap='Oranges')
                       
    # 4. New HTMLs (Legit vs Phishing)
    generate_wordcloud(df_new[df_new['category'] == 'ham']['html_clean'], 
                       "2026 Legitimate HTML Text", os.path.join(out_dir, "new_legit_html.png"), colormap='Blues')
    generate_wordcloud(df_new[df_new['category'] == 'spam']['html_clean'], 
                       "2026 Phishing HTML Text", os.path.join(out_dir, "new_phish_html.png"), colormap='Oranges')

    print("WordClouds successfully generated!")

if __name__ == "__main__":
    main()
