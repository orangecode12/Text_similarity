#import libraries
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import sys


#load model
#model  = SentenceTransformer("bert-base-uncased")
model  = SentenceTransformer('all-MiniLM-L6-v2')

# Define text input
def get_text(text_url):
     with open(text_url, 'r') as file:
        if not file.read(1):
            print("File you provide is empty")
            sys.exit(1)
        text_data = file.read()
        return text_data
     
# Generate n-grams from a given text
def generate_ngrams(text, n):
    words = text.split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    return ngrams
     
# Check if there are enough arguments passed
if len(sys.argv) < 2:
    print("Pass the file for analysis")
    sys.exit(1)

input_url = sys.argv[1]
input_data = get_text(input_url)

# Load Standartised terms
std_terms_url = "Standardised terms.csv"
std_terms_data = pd.read_csv(std_terms_url, sep="\t", header=None)

# Convert the single column in the DataFrame into a list of strings
std_terms_list = std_terms_data.iloc[:, 0].tolist()


# Generate 2-grams, 3-grams, 4-grams from a given text
all_ngrams = []
for n in range(2, 17): 
    all_ngrams.extend(generate_ngrams(input_data, n))


# Define threshold
#threshold = 0.77  #BERT
threshold = 0.6   #MiniLM

#Find phrases in input_text that are semantically similar to any of the standardised phrases
similarities = []

#Compute embedding for both lists
embeddings1 = model.encode(all_ngrams, convert_to_tensor=True)
embeddings2 = model.encode(std_terms_list, convert_to_tensor=True)

# Calculate cosine similarities using broadcasting
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

# Convert tensor to numpy for easier indexing and manipulation
cosine_scores = cosine_scores.numpy()


for i, sent1 in enumerate(all_ngrams):
    for j, sent2 in enumerate(std_terms_list):
            if cosine_scores[i][j] > threshold:
                similarities.append((sent1, sent2, cosine_scores[i][j]))
    
# Write the results to a file
with open('results.txt', 'w') as f:
    sorted_similarities = sorted(similarities, key=lambda x: x[2],  reverse=True)
    for similar in sorted_similarities:
        f.write(f"Cosine similarity between \"{similar[0]}\" and \"{similar[1]}\" is: {similar[2]:.4f}\n")
        print(f"Cosine similarity between \"{similar[0]}\" and \"{similar[1]}\" is: {similar[2]:.4f}")
