def create_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    print(f"Total words: {len(words)}")
    print(f"Chunk size: {chunk_size} words, Overlap: {overlap} words")
    
    start = 0
    chunk_id = 0
    
    while start < len(words):
        # Get chunk end position
        end = min(start + chunk_size, len(words))
        
        # Extract chunk words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        # Create chunk with metadata
        chunk = {
            "id": f"chunk_{chunk_id:03d}",
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end - 1
        }
        
        chunks.append(chunk)
        print(f"Created {chunk['id']}: {chunk['word_count']} words")
        
        # Move start position (with overlap)
        start = end - overlap
        chunk_id += 1
        
        # Prevent infinite loop
        if end >= len(words):
            break
    
    return chunks

if __name__ == "__main__":
    # Test with sample text
    sample_text = "This is a test document. " * 100  # 500 words
    chunks = create_chunks(sample_text, chunk_size=50, overlap=10)
    
    print(f"\nCreated {len(chunks)} chunks")
    print(f"First chunk preview: {chunks[0]['text'][:100]}...")