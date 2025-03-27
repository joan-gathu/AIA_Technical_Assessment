import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set Seaborn style
sns.set_style("whitegrid")

# Define color palette
sentiment_palette = {"positive": "green", "negative": "red", "neutral": "blue"}

def plot_sentiment_distribution(df, save_path="sentiment_distribution.png"):
    """
    Plots and saves the sentiment distribution as a bar chart.

    Args:
        df (pd.DataFrame): Dataframe containing 'sentiment' column.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x="sentiment", data=df, palette=sentiment_palette)
    plt.title("Sentiment Distribution", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.savefig(save_path)
    plt.close()
    print(f"Sentiment distribution saved to {save_path}")


def plot_sentiment_trends(df, save_path="sentiment_trends.png"):
    """
    Plots and saves sentiment trends over time.

    Args:
        df (pd.DataFrame): Dataframe containing 'date' and 'sentiment' columns.
        save_path (str): File path to save the plot.
    """
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(12, 6))

    # Group data by date and sentiment, then count occurrences
    sentiment_trend = df.groupby([df["date"].dt.to_period("M"), "sentiment"]).size().unstack().fillna(0)

    # Plot time-series data
    sentiment_trend.plot(kind="line", marker="o", figsize=(12, 6), color=["green", "red", "blue"])
    plt.title("Sentiment Trends Over Time", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Review Count", fontsize=12)
    plt.legend(title="Sentiment")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Sentiment trends saved to {save_path}")


def plot_category_sentiment(df, save_path="category_sentiment.png"):
    """
    Plots and saves sentiment distribution across product categories.

    Args:
        df (pd.DataFrame): Dataframe containing 'category' and 'sentiment' columns.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(14, 6))

    # Count sentiment occurrences per category
    category_sentiment = df.groupby(["category", "sentiment"]).size().unstack().fillna(0)

    # Plot stacked bar chart
    category_sentiment.plot(kind="bar", stacked=True, figsize=(14, 6), color=["green", "red", "blue"])
    plt.title("Sentiment Distribution Across Product Categories", fontsize=14)
    plt.xlabel("Product Category", fontsize=12)
    plt.ylabel("Review Count", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    plt.savefig(save_path)
    plt.close()
    print(f"Category sentiment distribution saved to {save_path}")


if __name__ == "__main__":
    # Use absolute path for better file handling
    file_path = os.path.join(os.getcwd(), "data", "product_reviews.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        plot_sentiment_distribution(df, "sentiment_distribution.png")
        plot_sentiment_trends(df, "sentiment_trends.png")
        plot_category_sentiment(df, "category_sentiment.png")
    else:
        print(f"Error: File not found at {file_path}")