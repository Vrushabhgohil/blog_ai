import os
import json
import re
from typing import List, Optional
from io import BytesIO

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from fastapi import UploadFile
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
import requests

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX_NAME)


def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """Extract text from PDF pages."""
    return "".join(page.extract_text() or "" for page in PdfReader(pdf_file).pages)


def extract_text(file: UploadFile) -> str:
    """Extract text from supported file types."""
    content = file.file.read()
    if file.filename.endswith(".pdf"):
        return extract_text_from_pdf(BytesIO(content))
    return content.decode("utf-8")


def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """Split text into chunks based on sentence length limit."""
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_length:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks


def generate_outline_prompt(title: str, language: str, context: Optional[str]) -> str:
    return f"""
        You are a professional multilingual blogger and SEO expert.

        Your task is to generate 7–8 blog post outlines based on the **core context** and topic.

        ## Context (High Priority)
        {context}

        ## Blog Title
        {title}

        ## Language
        {language}

        ## Instructions:
        - Each outline max 12 words, specific, SEO-relevant.
        - Return only a JSON array of outlines in {language}.
        """


def generate_blog_prompt(title: str, language: str, outlines: List[str], context: Optional[str]) -> str:
    outlines_str = "\n".join(f"- {o}" for o in outlines)
    return f"""
        You are an expert blog writer and SEO strategist fluent in {language}.

        Your task is to write a comprehensive, well-researched, and engaging **5000-word blog post** strictly in {language}, based on the details provided below.

        ## Context (High Priority)
        {context}

        ## Blog Title
        {title}

        ## Structure
        Use the following outlines as main sections (H2s). Expand each into 800–900 words:
        {outlines_str}

        ## Content Guidelines:
        - Total length: ~5000 words (balanced across all sections).
        - Language: Use {language} only, no English unless it's a quote, stat, or unavoidable reference.
        - Tone: Natural, professional, and engaging.
        - Style: Informative, storytelling where appropriate, and SEO-optimized.
        - Formatting: Use markdown headers (## for H2, ### for H3), bullet points, and short paragraphs for readability.
        - Content must be accurate, verifiable, and free from hallucinations. Use examples, real-world references, and reliable data where relevant.
        - Ensure smooth transitions between sections for a cohesive reading experience.

        ## Ending Requirements:
        At the end of the blog, include the following:
        - **Meta Description**: A concise summary (max 160 characters).
        - **SEO-Friendly URL Slug**: Based on the blog title and main keywords.
        - **SEO Tags**: 3 to 5 relevant keyword-based tags.
        - **Categories**: 2 to 3 relevant blog categories.
        - **Internal Linking Suggestions**: 1 to 2 ideas for linking to related internal content.
        """


def clean_html_response(raw: str) -> str:
    """Remove markdown or code formatting."""
    return raw.replace("```", "").replace("html", "").strip()


def answer_generator(
    title: str,
    language: str,
    outlines: Optional[List[str]] = None,
    mode: str = "blog",
    context: Optional[str] = "",
) -> str | List[str]:
    """Generate blog content or outlines using GROQ."""
    client = Groq(api_key=GROQ_API_KEY)

    prompt = generate_outline_prompt(title, language, context) if mode == "outline" \
        else generate_blog_prompt(title, language, outlines or [], context)

    system_prompt = "You are a helpful assistant. Return a JSON array of short outline titles." if mode == "outline" \
        else "You are a helpful assistant that returns blog content based on context and outlines."

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    content = response.choices[0].message.content.strip()

    if mode == "outline":
        try:
            json_block = re.search(
                r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", content)
            if json_block:
                return json.loads(json_block.group(1))

            inline_json = re.search(r"\[[\s\S]*?\]", content)
            if inline_json:
                return json.loads(inline_json.group(0))

            print("No JSON array found in outline response:", content)
            return []

        except json.JSONDecodeError:
            print("Outline JSON decoding error:", content)
            return []

    return clean_html_response(content)


def fetch_search_data(topic: str) -> str:
    """Fetch snippet context from Serper API based on a topic."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    payload = {"q": topic, "num": 5}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("Failed to fetch context from Serper API:", str(e))
        return ""

    return " ".join(r.get("snippet", "") for r in data.get("organic", []))
