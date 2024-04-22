import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize



# Load the CSV file
data = pd.read_csv('Fairness Taxonomies.csv')

# Display the first few rows of the dataframe along with column names
data.head(), data.columns

# Renaming the columns for better readability
data.rename(columns={'Types of bias': 'Level 1', 'Unnamed: 3': 'Level 2', 'Unnamed: 4': 'Level 3'}, inplace=True)

# Display the updated column names and the first few rows to confirm changes
data.head(), data.columns


# Consolidate all definitions into a single string for the word cloud
all_definitions = data['Definition'].dropna().str.cat(sep=' ')

# Add custom stopwords
custom_stopwords = {'and', 'no', 'yes', 'or'}

# Update the stopwords set with custom stopwords
stopwords = STOPWORDS.union(custom_stopwords)

# Function to generate and display a word cloud with stopwords removed
def generate_wordcloud_and_print_common_words(text, title, stopwords):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Get frequencies of words and sort them by frequency in descending order
    word_frequencies = wordcloud.words_
    common_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)[:30]

    return common_words


# Generate the word cloud and get the 30 most common words
common_words = generate_wordcloud_and_print_common_words(all_definitions, "All Levels - Combined Bias Types", stopwords)
common_words


# Function to generate word clouds for each level with respective types of biases and definitions
def generate_level_specific_wordclouds(data, levels, stopwords):
    for level in levels:
        # Filter relevant data for the level
        level_data = data[data[level] != '-'][[level, 'Definition']].dropna()

        # Concatenate all definitions in the level
        level_definitions = level_data['Definition'].str.cat(sep=' ')

        # Generate and display the word cloud for the level
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(
            level_definitions)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for {level}")
        plt.show()


# Generate word clouds for each level
generate_level_specific_wordclouds(data, ['Level 1', 'Level 2', 'Level 3'], stopwords)
