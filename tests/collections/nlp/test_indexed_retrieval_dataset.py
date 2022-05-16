# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import string

import numpy as np
import pytest

from nemo.collections.common.tokenizers.column_coder import ColumnCodes
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
import torch
from nemo.collections.nlp.data.language_modeling.megatron.indexed_retrieval_dataset import MMapRetrievalIndexedDataset, MMapRetrievalIndexedDatasetBuilder
import os


class TestTabularTokenizer:

    @pytest.mark.unit
    def test_index(self):
        chunk_size = 64
        sizes = np.array([128, 256], dtype=np.int32)
        dtype = np.int64
        itemsize = dtype().itemsize
        index_file = '/tmp/test.idx'
        try:
            with MMapRetrievalIndexedDataset.Index.writer(index_file, dtype) as index:
                index.write(sizes, chunk_size)

            index_load = MMapRetrievalIndexedDataset.Index(index_file)
            assert index_load.chunk_size == chunk_size
            assert np.array_equal(index_load.sizes, sizes)
            assert np.array_equal(index_load._chunk_id_start, np.array([0, sizes[0]/chunk_size], dtype=np.int64))
            assert np.array_equal(index_load._chunk_address, np.arange(0, sizes.sum()*itemsize, chunk_size * itemsize, dtype=np.int64))
            assert np.array_equal(index_load._pointers, np.array([0, sizes[0]*itemsize], dtype=np.int64))
            assert len(index_load._chunk_address) == index_load.num_chunks
        finally:
            os.remove(index_file)


    @pytest.mark.unit
    def test_create_data_index(self):
        chunk_size = 64
        sentence1 = torch.arange(0, 200, 2, dtype=torch.int64)
        sentence2 = torch.arange(1, 500, 2, dtype=torch.int64)
        pad_id = 0
        data_file = '/tmp/test'
        index_file = data_file+'.idx'
        bin_file = data_file+'.bin'
        try:
            builder = MMapRetrievalIndexedDatasetBuilder(bin_file, chunk_size, pad_id, False)
            builder.add_item(sentence1)
            builder.add_item(sentence2)
            builder.finalize(index_file)
            # load the data
            ds = MMapRetrievalIndexedDataset(data_file)
            print(ds.get(0))
            print(ds.get(1))
            print(ds[0:2])
        finally:
            os.remove(index_file)
            os.remove(bin_file)
