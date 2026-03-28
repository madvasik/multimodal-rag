from abc import ABC, abstractmethod
from typing import List
import torch

class BaseRetriever(ABC):

	@abstractmethod
	def retrieve(self, query: str | List[str], k: int = 2):
		"""Return nearest items from the index for each query."""
		pass

	@abstractmethod
	def _add_image_to_index(self, image_path: str) -> None:
		"""Add image embedding to the vector store."""
		pass

	@abstractmethod
	def embed_queries(self, query: str | List[str]) -> torch.Tensor:
		"""Embed one or more queries."""
		pass