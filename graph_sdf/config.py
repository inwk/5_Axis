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
    hidden_dim: int = 64
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.1
    action_embedding_dim: int = 64
    action_dropout: float = 0.1
    transition_decoder_layers: int = 3
    macro_class_count: int = len(MACRO_CLASS_TO_ID)
    tool_choice_count: int = len(TOOL_LIBRARY)
    centrality_vocab_size: int = 1024
    max_spatial_pos: int = 255
    sdf_channel_index: int = 6
    changed_face_threshold: float = 0.01

    def validate(self) -> None:
        """Raises an error when configuration values are invalid."""

        if self.hidden_dim % self.transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
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
