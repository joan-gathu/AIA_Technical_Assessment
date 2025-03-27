from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
import logging

from document_processing_pipeline import retrieve_similar_reviews

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_summary(category, reviews, max_reviews=5):
    """
    Generates a concise summary of product performance using GPT-4.

    Args:
        category (str): The product category (e.g., "Laptops").
        reviews (list): A list of customer review texts.
        max_reviews (int): Maximum number of reviews to include in the prompt.

    Returns:
        str: A summary of key praised features and common complaints.
    """
    if not reviews:
        return f"No reviews available for {category}."

    # Limit input reviews to avoid exceeding token limits
    selected_reviews = reviews[:max_reviews]

    prompt = f"""
    Summarize customer reviews for {category} products.
    Highlight key praised features and common complaints.

    Reviews:
    {selected_reviews}

    Summary:
    """

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert product reviewer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    return completion.choices[0].message.content


def generate_answer(query, top_n=3):
    """
    Retrieves relevant reviews and generates an answer using GPT-4.

    Args:
        query (str): User's question.
        top_n (int): Number of similar reviews to retrieve.

    Returns:
        dict: A dictionary with the question, relevant reviews, and GPT-generated answer.
    """
    # Retrieve similar reviews
    relevant_reviews = retrieve_similar_reviews(query, top_n=top_n)

    if not relevant_reviews:
        return {"Question": query, "Answer": "Sorry, I couldn't find relevant reviews to answer your question."}

    # Prepare context for GPT
    reviews_text = "\n".join(
        [f"- {rev['review_text']} (Rating: {rev['rating']}, Sentiment: {rev['sentiment']})" for rev in relevant_reviews])

    prompt = f"""
    You are an AI assistant answering user questions based on product reviews.

    Question: {query}

    Below are relevant customer reviews:
    {reviews_text}

    Based on these reviews, provide a helpful answer to the user's question.
    """

    # Generate response using GPT-4
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert product assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return {
        "Question": query,
        "Relevant Reviews": relevant_reviews,
        "Answer": completion.choices[0].message.content,
    }


def analyze_product_category(category, review_data, n_results=10):
    """
    Retrieves relevant reviews and summarizes common praises and issues using GPT-4.

    Args:
        category (str): Product category (e.g., "Laptops").
        review_data (pd.DataFrame): DataFrame containing product reviews.
        n_results (int): Number of reviews to retrieve.

    Returns:
        str: Summary of praised features and common complaints for the category.
    """
    # Retrieve relevant reviews
    retrieved_reviews = review_data[review_data["category"] == category].head(n_results)

    if retrieved_reviews.empty:
        return f"No reviews found for {category}."

    # Extract review text
    review_texts = retrieved_reviews["review_text"].tolist()

    prompt = f"""
    Summarize the key praised features and common complaints for {category} products
    based on the following reviews:

    {review_texts}

    Provide a structured response highlighting both praises and issues.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You analyze product reviews to extract common praises and complaints."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    # Use absolute path for better file handling
    file_path = os.path.join(os.getcwd(), "data", "product_reviews.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Generate summaries for each category
        category_summaries = df.groupby("category")["review_text"].apply(lambda x: generate_summary(x.name, x.tolist()))
        print("\n Product Performance Summaries:\n", category_summaries)

        # Q&A System
        question = "How is the battery life on smartphones?"
        answer = generate_answer(question, top_n=2)
        print("\n Answer to User Question:\n", answer)

        # Analyze common issues and praised features
        category_analysis = analyze_product_category("Laptops", df)
        print("\n Laptops - Praised Features & Issues:\n", category_analysis)
    else:
        print(f"Error: File not found at {file_path}")
