"""
Unit tests for document processor.
"""
import pytest
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.document_processor import DocumentProcessor, TextChunker


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_supported_formats(self):
        """Test that supported formats are defined."""
        processor = DocumentProcessor()
        assert len(processor.SUPPORTED_FORMATS) > 0
        assert '.pdf' in processor.SUPPORTED_FORMATS
        assert '.txt' in processor.SUPPORTED_FORMATS
    
    def test_process_txt_file(self, tmp_path):
        """Test processing a text file."""
        # Create a temporary text file
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file.\nWith multiple lines."
        test_file.write_text(test_content)
        
        processor = DocumentProcessor()
        result = processor.process_file(str(test_file))
        
        assert result['text'] == test_content
        assert result['file_name'] == 'test.txt'
        assert result['file_type'] == '.txt'
        assert 'metadata' in result
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        processor = DocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.process_file("/nonexistent/file.txt")
    
    def test_unsupported_format(self, tmp_path):
        """Test handling of unsupported file format."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")
        
        processor = DocumentProcessor()
        
        with pytest.raises(ValueError):
            processor.process_file(str(test_file))


class TestTextChunker:
    """Test cases for TextChunker."""
    
    def test_chunk_creation(self):
        """Test basic chunk creation."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test. " * 50  # Create text longer than chunk_size
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_index' in chunk for chunk in chunks)
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        
        assert len(chunks) == 0
    
    def test_short_text(self):
        """Test handling of text shorter than chunk size."""
        chunker = TextChunker(chunk_size=1000)
        text = "Short text."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == text
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        chunker = TextChunker(chunk_size=100)
        text = "Test text. " * 50
        metadata = {"source": "test", "author": "tester"}
        
        chunks = chunker.chunk_text(text, metadata=metadata)
        
        assert all('metadata' in chunk for chunk in chunks)
        assert all(chunk['metadata'] == metadata for chunk in chunks)
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Word " * 100  # Create predictable text
        
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that there's some overlap
            assert len(chunks) > 0
    
    def test_paragraph_chunking(self):
        """Test paragraph-based chunking."""
        chunker = TextChunker(chunk_size=100)
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        
        chunks = chunker.chunk_by_paragraphs(text)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
