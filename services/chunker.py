
import re

def split_text_into_chunks(text, max_len=600, overlap=100):
    """
    Split text into chunks of roughly max_len characters, with `overlap` characters
    of context preserved between chunks.
    
    This is a simplified equivalent of a RecursiveCharacterTextSplitter.
    It tries to split by paragraphs, then sentences, then spaces.
    """
    if not text:
        return []

    # naive list of separators to try in order
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    final_chunks = []
    
    # We will process a queue of text blocks
    # Initially, it's just the whole text
    # If a block is too big, we split it by the first separator that works
    
    def _split(text_block):
        if len(text_block) <= max_len:
            return [text_block]
        
        # Try finding a separator that gives us balanced splits
        for sep in separators:
            if sep == "":
                # Fallback: strict character slicing
                # We can't split by separator, so we just take the first max_len chars
                # But we need to be careful to loop.
                # Actually, simpler to just force slice
                chunks = []
                start = 0
                while start < len(text_block):
                    end = min(start + max_len, len(text_block))
                    chunks.append(text_block[start:end])
                    start = end - overlap # move back for overlap
                    if start >= len(text_block): break # avoid infinite loop if overlap >= max_len
                    # safeguard
                    if start < 0: start = 0
                return chunks

            # Check if this separator actually exists in the block
            if sep in text_block:
                parts = text_block.split(sep)
                # Now merge parts back until they fill a chunk
                merged = []
                current_chunk = ""
                
                for p in parts:
                    prefix = sep if current_chunk else "" # add separator back if inside chunk
                    candidate = current_chunk + prefix + p
                    
                    if len(candidate) <= max_len:
                        current_chunk = candidate
                    else:
                        if current_chunk:
                            merged.append(current_chunk)
                        
                        # Start new chunk
                        # Ideally we want overlap.
                        # Overlap is tricky in this simple accumulation loop.
                        # Simple approach: keep the last 'overlap' chars of current_chunk 
                        # and prepend to the next one? 
                        # For simplicity in this bespoke function, let's just push current_chunk
                        # and start new with 'p'. 
                        # To support proper overlap, we need a sliding window over the *tokens* or *chars*.
                        
                        current_chunk = p
                
                if current_chunk:
                    merged.append(current_chunk)
                    
                return merged
        return [text_block]

    # The above recursive-ish logic is hard to perfect in one go without the class
    # Let's use a simpler sliding window approach that is robust.
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + max_len
        
        # If we are not at the end, try to find a natural break point (space or newline)
        # to avoid cutting words in half
        if end < text_len:
            # Look for the last space/dot within the range [start+max_len*0.8, end]
            # so we don't shrink chunk too much.
            boundary_search_start = max(start, int(end - (max_len * 0.2)))
            slice_to_search = text[boundary_search_start:end]
            
            # Find last newline or space
            last_space = -1
            for sep in ["\n", ". ", " "]:
                idx = slice_to_search.rfind(sep)
                if idx != -1:
                    last_space = boundary_search_start + idx + len(sep) # include the separator in this chunk
                    break
            
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start pointer
        # We want the next chunk to start `difference` characters back from `end`
        # actually: next_start = end - overlap
        start = end - overlap
        
        # Safety check to ensure progress
        if start <= (end - max_len): # deadlock check
             start = end 
        
        # Initial start cannot be negative
        if start < 0:
            start = 0
            
    return chunks
