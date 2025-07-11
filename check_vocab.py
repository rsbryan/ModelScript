import pandas as pd
import re
import unicodedata

df = pd.read_csv('data/language_data.csv')

def normalise(text):
    # lower-case + strip accents so 'español' → 'espanol'
    text = unicodedata.normalize('NFD', text.lower())
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return re.sub(r'[^a-zñáéíóúü ]+', ' ', text)

vocab = set(w for t in df['text'].map(normalise) for w in t.split())
print('unique-tokens:', len(vocab))
print('sample vocab:', sorted(list(vocab))[:20])
print('spanish accented:', [w for w in vocab if any(c in w for c in 'ñáéíóúü')][:10])
print('original vs normalized examples:')
for i in range(5):
    orig = df['text'].iloc[i + 20]  # Some Spanish examples
    norm = normalise(orig)
    print(f'  "{orig}" → "{norm}"')