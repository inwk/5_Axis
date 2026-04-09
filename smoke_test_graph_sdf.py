"""Simple smoke test for the Graph-SDF planner + transition package."""

import torch

from graph_sdf import GraphSdfModelConfig, GraphSdfPlanningModel


def run_smoke_test() -> None:
    """Runs a minimal planner and transition forward pass on random tensors."""

    cfg = GraphSdfModelConfig()
    model = GraphSdfPlanningModel(cfg)

    batch_size = 2
    state_points = torch.randn(batch_size, cfg.num_nodes, cfg.points_per_node, cfg.point_feature_dim)
    node_process_state = torch.randint(0, 2, (batch_size, cfg.num_nodes, cfg.node_process_feature_dim)).float()
    node_centrality = torch.randint(0, 8, (batch_size, cfg.num_nodes))
    spatial_pos = torch.randint(0, 32, (batch_size, cfg.num_nodes, cfg.num_nodes))
    face_area = torch.rand(batch_size, cfg.num_nodes, cfg.face_area_feature_dim)
    node_mask = torch.zeros(batch_size, cfg.num_nodes, dtype=torch.bool)
    macro_class_id = torch.randint(0, cfg.macro_class_count, (batch_size,))
    tool_choice_id = torch.randint(0, cfg.tool_choice_count, (batch_size,))
    strategy_id = torch.randint(0, cfg.strategy_count, (batch_size,))
    target_node_id = torch.randint(0, cfg.num_nodes, (batch_size,))

    outputs = model(
        state_points=state_points,
        node_process_state=node_process_state,
        node_centrality=node_centrality,
        spatial_pos=spatial_pos,
        face_area=face_area,
        node_mask=node_mask,
        target_macro_class=macro_class_id,
        target_tool_choice=tool_choice_id,
        target_strategy=strategy_id,
        target_node_id=target_node_id,
        use_teacher_forcing_action=True,
        run_transition=True,
    )
    print("[Planner] macro_class_logits:", tuple(outputs["macro_class_logits"].shape))
    print("[Planner] target_node_logits:", tuple(outputs["target_node_logits"].shape))
    print("[Planner] tool_choice_logits:", tuple(outputs["tool_choice_logits"].shape))
    print("[Planner] strategy_logits:", tuple(outputs["strategy_logits"].shape))
    print("[Planner] pred_axis_from_target:", tuple(outputs["pred_axis_from_target"].shape))
    print("[Planner] pred_axis_mode:", tuple(outputs["pred_axis_mode"].shape))
    print("[Transition] pred_next_node_sdf:", tuple(outputs["pred_next_node_sdf"].shape))
    print("[Transition] pred_next_point_sdf:", tuple(outputs["pred_next_point_sdf"].shape))
    print("[Transition] pred_changed_logits:", tuple(outputs["pred_changed_logits"].shape))


if __name__ == "__main__":
    run_smoke_test()
