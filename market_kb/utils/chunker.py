from typing import List, Dict, Any
import re

class SemanticChunker():
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        text = re.sub(r'\n{3,}', '\n\n', text)

        splits = [
            r'(?<=\.)\s*\n\n(?=[A-Z])',
            r'(?<=\n)\s*#{1,6}\s+[A-Z]',
            r'(?<=\n)[A-Z][^a-z]*?:\s*\n',
            r'(?<=\n\n)\s*(?:[-•\*]|\d+\.)\s+',
            r'(?<=\.)\s{2,}(?=[A-Z])',
            r'(?<=[\.\?\!])\s+(?=[A-Z])'
        ]

        pattern = '|'.join(splits)
        segments = re.split(pattern, text.strip())

        chunks = []
        current_chunk = []
        current_length = 0
        current_pos = 0
        chunk_start = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            if current_length + len(segment) > self.max_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start": chunk_start,
                    "end": chunk_start + len(chunk_text)
                })
                current_chunk = []
                current_length = 0
                chunk_start = current_pos

            current_chunk.append(segment)
            current_length += len(segment)
            current_pos += len(segment) + 1

            if current_length >= self.min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start": chunk_start,
                    "end": chunk_start + len(chunk_text)
                })
                current_chunk = []
                current_length = 0
                chunk_start = current_pos

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "start": chunk_start,
                "end": chunk_start + len(chunk_text)
            })

        return chunks