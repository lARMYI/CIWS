"""Document ingestion: extraction, chunking, embedding, corpus search."""

from .chunk import TextChunk, chunk_code, chunk_text
from .extractors import Extracted, extract, extract_url, supported_extensions
from .pipeline import (
    delete_document,
    get_chunks,
    get_document,
    ingest_directory,
    ingest_file,
    ingest_text,
    ingest_url,
    list_documents,
    reindex,
    search_corpus,
    stats,
    summarize_document,
)

__all__ = [
    "TextChunk", "Extracted", "chunk_code", "chunk_text", "extract", "extract_url",
    "supported_extensions", "delete_document", "get_chunks", "get_document",
    "ingest_directory", "ingest_file", "ingest_text", "ingest_url", "list_documents",
    "reindex", "search_corpus", "stats", "summarize_document",
]
