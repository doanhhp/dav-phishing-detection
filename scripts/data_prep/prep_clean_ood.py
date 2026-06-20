import pandas as pd
import numpy as np

def clean_dataset():
    url_file = 'data/raw/OOD_URL.xlsx'
    html_file = 'data/raw/OOD_html.xlsx'
    
    print("Loading datasets...")
    try:
        df_url = pd.read_excel(url_file, engine='calamine')
        df_html = pd.read_excel(html_file, engine='calamine')
    except:
        df_url = pd.read_excel(url_file)
        df_html = pd.read_excel(html_file)

    initial_len = len(df_url)
    print(f"Initial Dataset Size: {initial_len} records")

    # Combine temporarily for synchronized filtering
    df = pd.DataFrame({
        'Category': df_url['Category'],
        'URL': df_url['Data'],
        'HTML': df_html['Data']
    })

    # 1. Drop NaN values
    df = df.dropna()
    print(f"After dropping NaNs: {len(df)} records")

    # 2. Drop duplicates based on URL
    df = df.drop_duplicates(subset=['URL'], keep='first')
    print(f"After dropping duplicate URLs: {len(df)} records")

    # 3. Drop suspiciously short HTML (less than 150 characters)
    # Often indicates "404 Not Found" or "Forbidden" pages without actual content
    df['HTML_str'] = df['HTML'].astype(str)
    df = df[df['HTML_str'].str.len() > 150]
    print(f"After dropping very short HTML (<150 chars): {len(df)} records")

    # 4. Drop common server errors
    error_signatures = [
        "404 Not Found", "403 Forbidden", "Access Denied", 
        "502 Bad Gateway", "Service Unavailable", "Not Acceptable!"
    ]
    
    # We only drop if the HTML is suspiciously short (e.g. < 2000 chars) AND contains the error, 
    # because some real sites might have these words somewhere deep in their code.
    mask = ~((df['HTML_str'].str.len() < 2000) & (df['HTML_str'].str.contains('|'.join(error_signatures), case=False, na=False)))
    df = df[mask]
    
    print(f"After removing server error pages: {len(df)} records")
    
    # 5. Check Class Balance
    print("\nFinal Class Distribution:")
    print(df['Category'].value_counts())

    print("\nSaving cleaned datasets...")
    # Reconstruct original format
    final_url = pd.DataFrame({
        'Category': df['Category'],
        'Data': df['URL']
    })
    
    final_html = pd.DataFrame({
        'Category': df['Category'],
        'Data': df['HTML']
    })

    final_url.to_excel(url_file, index=False)
    final_html.to_excel(html_file, index=False)
    
    print(f"Successfully cleaned! Removed {initial_len - len(df)} bad/duplicate records.")

if __name__ == "__main__":
    clean_dataset()
