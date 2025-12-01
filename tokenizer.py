import collections
import re
from typing import Dict, List, Tuple

class CustomBPE:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.token_to_id: Dict[bytes, int] = {}
        self.id_to_token: Dict[int, bytes] = {}

    def _get_stats(self, tokens):
        counts = collections.defaultdict(int)
        for chunk in tokens:
            for i in range(len(chunk) - 1):
                counts[(chunk[i], chunk[i+1])] += 1
        return counts

    def merge_pair_in_corpus(self, chunks: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        merged_chunks = []
        for chunk in chunks:
            new_chunk = []
            i = 0
            while i < len(chunk):
                if i < len(chunk) - 1 and (chunk[i], chunk[i+1]) == pair:
                    new_chunk.append(new_id)
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            merged_chunks.append(new_chunk)
        return merged_chunks
    
    def train(self, text_corpus: str):
        byte_tokens = [list(c.encode("utf-8")) for c in re.split(r"(\s+)", text_corpus) if c]
        
        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.id_to_token = {i: bytes([i]) for i in range(256)}
        next_id = 256

        while len(self.token_to_id) < self.vocab_size:
              stats = self._get_stats(byte_tokens)
  
              if not stats:
                  print("Error!")
                  break
               
              best_pair = max(stats, key=stats.get)
              new_token = best_pair[0] + best_pair[1]

              self.merges[best_pair] = next_id
              self.token_to_id[new_token] = next_id
              self.id_to_token[next_id] = new_token

              byte_tokens = self._merge_pair_in_corpus(byte_tokens, best_pair, next_id)
              next_id += 1
              print("Finished!")

    def encode(self, text: str) -> List[int]:
        tokens = [list(c.encode("utf-8")) for c in re.split(r"(\s+)", text) if c]

        for pair, new_id in self.merges.items():
            new_tokens = []
            for token_list in tokens:
                new_tokens.append(self._merge_pair_in_corpus([token_list], pair, new_id)[0])
            tokens = new_tokens

        final_ids = []
        for token_list in tokens:
            for token in token_list:
                final_ids.append(self.token_to_id.get(token, self.token_to_id.get(b"<unk>")))
        return final_ids
