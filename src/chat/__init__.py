"""Chat / conversation persistence layer. Kept completely separate from
the Chroma vector store — this package owns SQLite, that one owns
vectors, and neither is ever a substitute for the other."""
