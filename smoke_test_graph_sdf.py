"""Simple smoke test for the Graph-SDF planner package."""

import torch

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel


def run_smoke_test() -> None:
    """Runs a minimal planner forward pass on random tensors."""

    cfg = GraphSdfModelConfig()
    model = GraphSdfPlanningModel(cfg)

    batch_size = 2
    state_points = torch.randn(batch_size, cfg.num_nodes, cfg.points_per_node, cfg.point_feature_dim)
    node_process_state = torch.randint(0, 2, (batch_size, cfg.num_nodes, cfg.node_process_feature_dim)).float()
    node_mask = torch.zeros(batch_size, cfg.num_nodes, dtype=torch.bool)

    outputs = model.forward_process_planner(
        state_points=state_points,
        node_process_state=node_process_state,
        node_mask=node_mask,
    )
    print("[Planner] macro_class_logits:", tuple(outputs["macro_class_logits"].shape))
    print("[Planner] target_node_logits:", tuple(outputs["target_node_logits"].shape))
    print("[Planner] tool_choice_logits:", tuple(outputs["tool_choice_logits"].shape))
    print("[Planner] pred_axis_from_target:", tuple(outputs["pred_axis_from_target"].shape))


if __name__ == "__main__":
    run_smoke_test()
