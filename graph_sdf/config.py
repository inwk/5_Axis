"""Configuration dataclass for the Graph-SDF process skeleton planner."""

from dataclasses import dataclass

from .schema import MACRO_CLASS_TO_ID, TOOL_LIBRARY


@dataclass(frozen=True)
class GraphSdfModelConfig:
    """Stores all planner and encoder hyperparameters in one place."""

    num_nodes: int = 512
    points_per_node: int = 100
    point_feature_dim: int = 7
    node_process_feature_dim: int = 2
    global_process_feature_dim: int = 11
    face_area_feature_dim: int = 1
    face_type_vocab_size: int = 32
    hidden_dim: int = 128                 # 64 → 128: octree decoder needs more capacity
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    action_embedding_dim: int = 128       # must match hidden_dim
    action_dropout: float = 0.1
    transition_decoder_layers: int = 3
    macro_class_count: int = len(MACRO_CLASS_TO_ID)
    tool_choice_count: int = len(TOOL_LIBRARY)
    centrality_vocab_size: int = 1024
    max_spatial_pos: int = 255
    sdf_channel_index: int = 6
    changed_face_threshold: float = 0.01

    # ── Decoder selection ────────────────────────────────────────────────
    # use_sdf_decoder=False: per-face SDF head is skipped (ActionEmbedding
    # inside ShapeTransitionHead is still used to build action_context).
    use_sdf_decoder: bool = False
    use_octree_decoder: bool = True
    use_sdf_query_decoder: bool = True

    # ── Octree decoder ────────────────────────────────────────────────────
    # Data collection parameters (used by collect_axis_dataset.py / CAM/measurements.py)
    #   octree_coarse_depth : build uniform grid at this depth first  (8^d cells)
    #   octree_fine_depth   : refine boundary cells down to this depth
    # Network parameters
    #   octree_query_nodes  : K nodes sampled from the stored octree per training step
    #   octree_fourier_bands: L in the scale-aware Fourier positional encoding
    #                         input dim = 4*(1 + 2*L)  [xyz + norm_depth, each with L bands]
    #   octree_cross_attn_heads : heads for octree-node → face-node cross-attention
    #   octree_decoder_layers   : number of CrossAttn + FiLM residual blocks
    octree_coarse_depth: int = 3          # 8³ = 512 coarse cells checked (NX API calls)
    octree_fine_depth: int = 5            # refine boundary cells → effective 32³ at surface
    octree_query_nodes: int = 4096        # K nodes per training step (sampled from stored octree)
    sdf_query_nodes: int = 32768           # Q query points per training step for TSDF transition
    octree_fourier_bands: int = 6         # positional encoding bandwidth
    octree_cross_attn_heads: int = 4      # must divide hidden_dim
    octree_decoder_layers: int = 3
    octree_dropout: float = 0.1

    def validate(self) -> None:
        """Raises an error when configuration values are invalid."""

        if self.hidden_dim % self.transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if self.hidden_dim % self.octree_cross_attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by octree_cross_attn_heads")
        if self.num_nodes <= 0 or self.points_per_node <= 0:
            raise ValueError("num_nodes and points_per_node must be positive")
        if self.point_feature_dim <= self.sdf_channel_index:
            raise ValueError("sdf_channel_index must reference an existing point feature")
        if self.node_process_feature_dim < 0:
            raise ValueError("node_process_feature_dim must be non-negative")
        if self.global_process_feature_dim < 0:
            raise ValueError("global_process_feature_dim must be non-negative")
        if self.face_area_feature_dim < 0:
            raise ValueError("face_area_feature_dim must be non-negative")
        if self.face_type_vocab_size <= 0:
            raise ValueError("face_type_vocab_size must be positive")
        if self.action_embedding_dim <= 0:
            raise ValueError("action_embedding_dim must be positive")
        if self.transition_decoder_layers <= 0:
            raise ValueError("transition_decoder_layers must be positive")
        if self.macro_class_count <= 1:
            raise ValueError("macro_class_count must be greater than 1")
        if self.tool_choice_count <= 0:
            raise ValueError("tool_choice_count must be positive")
        if self.centrality_vocab_size <= 0:
            raise ValueError("centrality_vocab_size must be positive")
        if self.max_spatial_pos <= 0:
            raise ValueError("max_spatial_pos must be positive")
        if self.octree_fine_depth < self.octree_coarse_depth:
            raise ValueError("octree_fine_depth must be >= octree_coarse_depth")
        if self.octree_query_nodes <= 0:
            raise ValueError("octree_query_nodes must be positive")
        if self.sdf_query_nodes <= 0:
            raise ValueError("sdf_query_nodes must be positive")
        if self.octree_fourier_bands <= 0:
            raise ValueError("octree_fourier_bands must be positive")
