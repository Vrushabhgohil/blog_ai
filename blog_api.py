from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os, uuid

from utils import (
    extract_text,
    chunk_text,
    embedding_model,
    answer_generator,
    index,
    fetch_search_data
)

from dotenv import load_dotenv
load_dotenv()

blog_router = APIRouter()

SUPPORTED_LANGUAGES = {"english", "hindi", "gujarati"}


class BlogData(BaseModel):
    title: str
    language: str = Field(default="English")


def generate_full_blog(title: str, language: str, context: str):
    """
    Generates blog outlines and content using the answer generator.
    """
    outlines = answer_generator(
        title=title,
        language=language,
        outlines=None,
        mode="outline",
        context=context
    )

    if not outlines:
        raise HTTPException(status_code=500, detail="Failed to generate outlines.")

    blog = answer_generator(
        title=title,
        language=language,
        outlines=outlines,
        mode="blog",
        context=context
    )

    return {
        "status": "success",
        "title": title,
        "outlines": outlines,
        "blog": blog
    }


@blog_router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Extracts and chunks text from the uploaded document,
    embeds it, and stores in vector DB.
    """
    try:
        text = extract_text(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    chunks = chunk_text(text)
    vectors = [
        {
            "id": str(uuid.uuid4()),
            "values": embedding_model.encode(chunk).tolist(),
            "metadata": {"text": chunk, "filename": file.filename}
        }
        for chunk in chunks
    ]

    index.upsert(vectors=vectors)
    return {"status": "success", "chunks_uploaded": len(vectors)}


@blog_router.post("/generate-blog")
def blog_generation(request: BlogData):
    """
    Generates a blog using embedded document context.
    """
    title = request.title.strip()
    language = request.language.lower().strip()

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Language not supported.")

    query_vector = embedding_model.encode(title).tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    context = "\n".join(match["metadata"]["text"] for match in results.get("matches", []))
    return generate_full_blog(title, language, context)


@blog_router.post("/v2/generate-blog")
def blog_generation_v2(request: BlogData):
    """
    Generates a blog using live search-based context.
    """
    title = request.title.strip()
    language = request.language.lower().strip()

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Language not supported.")

    context = fetch_search_data(title)
    return generate_full_blog(title, language, context)
