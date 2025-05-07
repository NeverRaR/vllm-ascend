#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType,
                                              MLAAttentionImpl)
from vllm.attention.backends.utils import CommonAttentionState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import direct_register_custom_op

from vllm.attention.backends.utils import CommonAttentionState, PAD_SLOT_ID
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm.logger import logger 
from vllm.config import get_current_vllm_config


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

class AscendMLAAttentionBackend(AscendAttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["AscendMLAAttentionBackendImpl"]:
        return AscendMLAAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

class AscendAttentionState(Enum):
    PrefillOnly = 0
    DecodeOnly = 1
    ChunkedPrefill = 2

@dataclass
class AscendMetadata:
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    block_tables: torch.Tensor
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    query_lens: torch.Tensor
    #seq_lens: torch.Tensor
    seq_lens: Optional[torch.Tensor] = None

    seq_lens_list: Optional[list[int]] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None
    # (num_tokens,). The indices of the token slots that input tokens will be
    # stored into. E.g., if `slot_mapping` is [35, 2, 17] and the block size
    # is 16, the three tokens are stored in the 3rd slot in block 2, 2nd slot
    # in block 0, and 1st slot in block 1, respectively.
    slot_mapping: torch.Tensor = None
    # TODO: Indicates whether there are only prefill requests.
    # FlashAttention can be used when there are only prefill requests.
    # FlashAttention has better performance than PageAtttention,
    # but it does not support decode requests.
    is_only_prefill: bool = False
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    attn_mask: Optional[torch.Tensor] = None
    input_positions: torch.Tensor =None

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0


class AscendAttentionMetadataBuilder:

    def __init__(self, runner):
        self.runner = runner

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch


    def _get_graph_runner_block_tables(
        self, num_seqs: int,
        block_tables: torch.Tensor) -> torch.Tensor:
    
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs
    
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            # 如果graph_block_tables是numpy数组，创建新的tensor
            graph_block_tables = torch.zeros(
                (max_batch_size, max_blocks),
                dtype=block_tables.dtype,
                device=block_tables.device
            )
        else:
            # 如果已经是tensor，直接使用
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device,
                dtype=block_tables.dtype
            )
    
        # 复制数据
        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            # 如果block数量在范围内，直接复制
            graph_block_tables[:num_seqs, :num_blocks] = block_tables[:num_seqs, :num_blocks]
        else:
            # 如果超出范围，只复制允许的最大数量
            graph_block_tables[:num_seqs, :max_blocks] = block_tables[:num_seqs, :max_blocks]
    
        return graph_block_tables



    def build(self, num_reqs, num_actual_tokens, max_query_len,
              common_prefix_len, graph_pad_size):
        block_tables = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        query_lens = self.runner.query_lens
        seq_lens = self.runner.seq_lens_cpu[:num_reqs]
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()
        attn_mask = self.runner.attn_mask
        attn_state = self.runner.attn_state

        use_torchair_graph = graph_pad_size != -1
        tokens_start = self._num_decode_tokens
        if self.runner.attn_state == AscendAttentionState.DecodeOnly:
            input_positions=input_positions[:self._num_decode_tokens]
        else:
            input_positions=input_positions[tokens_start:]
        seq_lens_list = []
        if use_torchair_graph and self.runner.attn_state == AscendAttentionState.DecodeOnly:
            num_seqs = len(seq_lens)
            if graph_pad_size != 0:
                pad_value = 1
                #pad_value = seq_lens.max().item()
                padded_seq_lens = seq_lens.tolist() + [pad_value] * graph_pad_size
            else:
                padded_seq_lens = seq_lens.tolist()

            seq_lens = torch.from_numpy(
                            np.array(padded_seq_lens).
                            astype(np.int32))

            seq_lens_list = seq_lens.tolist()
            #print("use_torchair_graph seq_lens:", seq_lens)
            padding = torch.full((graph_pad_size,), PAD_SLOT_ID, 
                           dtype=slot_mapping.dtype,
                           device=slot_mapping.device)
            slot_mapping = torch.cat([slot_mapping, padding])
            block_table_padding = torch.zeros((graph_pad_size,) + block_tables.shape[1:],
                       dtype=block_tables.dtype,
                       device=block_tables.device)
            block_tables = torch.cat([block_tables, block_table_padding], dim=0)
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, block_tables)

        attn_metadata = AscendMetadata(block_tables=block_tables,
                                       query_lens=query_lens,
                                       seq_lens=seq_lens,
                                       seq_lens_list=seq_lens_list,
                                       max_query_len=max_query_len,
                                       slot_mapping=slot_mapping,
                                       attn_mask=attn_mask,
                                       attn_state=attn_state,
                                       input_positions=input_positions,
                                       num_decodes=self._num_decodes,
                                       num_decode_tokens=self._num_decode_tokens,
                                       num_prefills=self._num_prefills,
                                       )
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.enable_graph_mode = False
        additional_config = get_current_vllm_config().additional_config
        if additional_config:
            self.enable_graph_mode = additional_config.get(
                "enable_graph_mode", False)


    def functional_reshape_and_cache(self, key, value, key_cache, value_cache, slot_indices):
        new_key = key.reshape(-1, key.size(1), key.size(-1))
        new_value = value.reshape(-1, value.size(1), value.size(-1))
    
        new_key_cache = key_cache.clone()
        new_value_cache = value_cache.clone()
    
        if slot_indices is not None:
            new_key_cache.index_copy_(0, slot_indices, new_key)
            new_value_cache.index_copy_(0, slot_indices, new_value)
    
        return new_key, new_value, new_key_cache, new_value_cache

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size,
                            num_kv_heads * head_size]
                    key_cache = [num_blocks, block_size,
                                num_kv_heads * head_size]
                    value_cache = [num_blocks, block_size,
                                    num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]

        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        if trace_flag and not self.enable_graph_mode:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)
        else:
            num_tokens = query.shape[0]
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size)
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if attn_type != AttentionType.DECODER:
                raise NotImplementedError("Encoder self-attention and "
                                          "encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()


            if self.enable_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if kv_cache.numel() > 0:
                    if self.key_cache is None:
                        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                    slots = attn_metadata.slot_mapping
                if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                
                    seq_lens = attn_metadata.seq_lens_list

                    # Reshape query for graph mode
                    query = query.view(num_tokens, self.num_heads, 1, self.head_size)

                    # Reshape query for graph mode
                    #query = query.view(num_tokens, self.num_heads, 1, -1)
                
                    block_tables = attn_metadata.block_tables
                    if block_tables is not None:
                        block_tables = block_tables.contiguous()

                    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                            query=query,
                            key=self.key_cache,
                            value=self.value_cache,
                            num_heads=self.num_heads,
                            num_key_value_heads=self.num_kv_heads,
                            input_layout="BNSD",
                            atten_mask=attn_metadata.attn_mask,
                            scale=self.scale,
                            antiquant_mode=0,
                            antiquant_scale=None,
                            block_table=block_tables,
                            block_size=self.key_cache.shape[1],
                            actual_seq_lengths_kv=seq_lens
                    )
                    output = attn_output.view(num_tokens, self.num_heads, self.head_size)
                    return output
                    
            if kv_cache.numel() > 0:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(key=key,
                                                 value=value,
                                                 key_cache=self.key_cache,
                                                 value_cache=self.value_cache,
                                                 slot_indices=slots)

            if hasattr(layer, 'quant_method'):
                # TODO: Add attr (num_prefills, prefill_metadata, decode_metadata) to AscendMetadata
                pass
            # V0-Style scheduler situation.
            elif attn_metadata.attn_state == AscendAttentionState.PrefillOnly:
                assert attn_metadata is not None
                assert attn_metadata.attn_mask is not None
                mask = attn_metadata.attn_mask
                torch_npu._npu_flash_attention(query=query,
                                               key=key,
                                               value=value,
                                               mask=mask,
                                               seq_len=attn_metadata.seq_lens,
                                               scale_value=self.scale,
                                               num_heads=self.num_heads,
                                               num_kv_heads=self.num_kv_heads,
                                               out=output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                block_tables = attn_metadata.block_tables
                logger.warning(f"kv cache shape: {kv_cache[0].shape}")
                logger.warning(f"block_size: {kv_cache[0].shape[1]}")
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
            # Normal V1 situation.
            else:
                # use chunked prefill for head size 192 scenario, like deepseek
                # paged_attention_splitfuse maybe crash at such scenario
                # TODO: vanilla path will be removed after the kernel support
                # head_size 192 scenario
                if self.head_size == 192:
                    cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
                    cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
                    cu_seqlen_q = torch.tensor(cu_seqlen_q, device="npu")
                    cu_seqlen_k = torch.tensor(cu_seqlen_k, device="npu")
                    cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
                    cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
                    max_seqlen_q = torch.max(attn_metadata.query_lens)
                    max_seqlen_k = torch.max(attn_metadata.seq_lens)
                    vanilla_chunked_prefill(output, query, self.key_cache,
                                            self.value_cache,
                                            attn_metadata.block_tables,
                                            cu_seqlen_q, cu_seqlen_k,
                                            max_seqlen_q, max_seqlen_k,
                                            self.scale, None, True)
                else:
                    # use paged attention
                    torch_npu._npu_paged_attention_splitfuse(
                        query=query,
                        key_cache=self.key_cache,
                        value_cache=self.value_cache,
                        mask=attn_metadata.attn_mask,
                        block_table=attn_metadata.block_tables,
                        seq_len=attn_metadata.query_lens,
                        context_lens=attn_metadata.seq_lens,
                        num_kv_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        scale_value=self.scale,
                        out=output)
        return output.view(num_tokens, self.hidden_size)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
