# import torch

# class SparseDenseIndexTranslator:
#   def __init__(
#     self,
#     sparse_index_type: torch.dtype,
#     dense_index_type: torch.dtype,
#     dense_index_begin: int,
#     dense_index_end: int,
#   ):
#     self.sparse_index_type = sparse_index_type
#     self.dense_index_type = dense_index_type
#     self.dense_index_begin = dense_index_begin
#     self.dense_index_end = dense_index_end
#     self.capacity = dense_index_end - dense_index_begin
#     self.keys = torch.