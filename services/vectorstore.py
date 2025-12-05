# services/vectorstore.py
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

import config
from services.embeddings import get_embeddings
from services.chunker import split_text_into_chunks

class KBManager:
    def __init__(self, root_dir: str = str(config.DATA_DIR)):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        # optional in-memory cache for loaded KB objects to help release indexes on delete
        self._loaded_kbs = {}

    def list_kbs(self):
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def create_kb(self, name):
        kb_dir = self.root / name
        kb_dir.mkdir(parents=True, exist_ok=True)
        meta_path = kb_dir / 'metadata.pkl'
        if not meta_path.exists():
            with open(meta_path, 'wb') as f:
                pickle.dump({'docs': []}, f)

    def delete_kb(self, name):
        """
        Robustly delete KB folder and clear any in-memory index handles.
        Raises RuntimeError with details on failure.
        """
        kb_dir = self.root / name
        if not kb_dir.exists():
            raise RuntimeError(f"KB folder does not exist: {kb_dir}")

        # Attempt to clear any in-memory cache & release references to FAISS indexes
        try:
            if name in self._loaded_kbs:
                kb_obj = self._loaded_kbs[name]
                try:
                    # try to release index reference
                    if hasattr(kb_obj, 'index'):
                        kb_obj.index = None
                except Exception:
                    pass
                try:
                    del self._loaded_kbs[name]
                except Exception:
                    pass
        except Exception:
            # non-fatal, proceed to disk removal
            pass

        # Now remove directory from disk
        try:
            shutil.rmtree(kb_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to delete KB '{name}' at {kb_dir}: {e}")

        return True

    def get_kb(self, name):
        kb = KB(name, self.root / name)
        # cache loaded instance to help manage memory / closures
        self._loaded_kbs[name] = kb
        return kb

    def add_transcript(self, kb_name, title, transcript):
        kb = self.get_kb(kb_name)
        kb.add_document(title, transcript)


class KB:
    def __init__(self, name, path: Path):
        self.name = name
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.path / 'index.faiss'
        self.meta_path = self.path / 'metadata.pkl'

        if self.meta_path.exists():
            with open(self.meta_path, 'rb') as f:
                try:
                    self.metadata = pickle.load(f)
                except Exception:
                    self.metadata = {'docs': []}
        else:
            self.metadata = {'docs': []}

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception:
                # if index corrupted, ignore and rebuild on next add
                self.index = None
        else:
            self.index = None

    def save(self):
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        if self.index is not None:
            try:
                faiss.write_index(self.index, str(self.index_path))
            except Exception:
                # ignore write errors for now
                pass

    def add_document(self, title, text):
        """
        Add a document to the KB. Handles embedding-dimension mismatches by rebuilding
        the FAISS index using the current embedding model if needed.
        """
        chunks = split_text_into_chunks(text)
        if not chunks:
            return

        # get embeddings for new chunks
        embs_new = get_embeddings(chunks)
        import numpy as _np
        vecs_new = _np.array(embs_new).astype('float32')
        new_dim = vecs_new.shape[1]

        # If no index, create new index and add
        if self.index is None:
            self.index = faiss.IndexFlatL2(new_dim)
            self.index.add(vecs_new)
            for c in chunks:
                self.metadata['docs'].append({'title': title, 'text': c})
            self.save()
            return

        # If index exists, check dimension compatibility
        try:
            existing_dim = self.index.d
        except Exception:
            # if index object doesn't expose .d, fallback to rebuilding
            existing_dim = None

        if existing_dim == new_dim:
            # append new vectors
            self.index.add(vecs_new)
            for c in chunks:
                self.metadata['docs'].append({'title': title, 'text': c})
            self.save()
            return

        # DIMENSION MISMATCH -> rebuild index by re-embedding all docs with current model
        try:
            existing_texts = [doc['text'] for doc in self.metadata.get('docs', [])]
            all_texts = existing_texts + chunks
            all_embs = get_embeddings(all_texts)
            all_vecs = _np.array(all_embs).astype('float32')
            dim = all_vecs.shape[1]

            # build new index
            new_index = faiss.IndexFlatL2(dim)
            new_index.add(all_vecs)

            # reconstruct metadata
            new_metadata = []
            for doc in self.metadata.get('docs', []):
                new_metadata.append({'title': doc.get('title'), 'text': doc.get('text')})
            for c in chunks:
                new_metadata.append({'title': title, 'text': c})

            # replace index + metadata
            self.index = new_index
            self.metadata['docs'] = new_metadata
            self.save()
            return

        except Exception as e:
            raise RuntimeError(
                "Failed to rebuild FAISS index after detecting embedding-dimension mismatch. "
                "Details: " + str(e) + ".\n"
                "As a fallback you can delete the KB folder and re-add documents: "
                f"rm -rf {self.path}"
            )

    def query(self, query_text, top_k=4):
        q_emb = get_embeddings([query_text])[0]
        q_vec = np.array(q_emb).astype('float32').reshape(1, -1)
        if self.index is None or (hasattr(self.index, 'ntotal') and self.index.ntotal == 0):
            return []
        D, I = self.index.search(q_vec, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            try:
                doc = self.metadata['docs'][idx]
            except Exception:
                doc = {'title': None, 'text': '[missing]'}
            results.append({'score': float(dist), 'text': doc.get('text'), 'title': doc.get('title')})
        return results
