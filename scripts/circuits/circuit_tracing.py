from pathlib import Path
import torch

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.utils import create_graph_files
from circuit_tracer.graph import prune_graph


class CircuitTracer:
    def __init__(self, model_name="Qwen/Qwen3-4B", transcoder_name="mwhanna/qwen3-4b-transcoders"):
        self.model_name = model_name
        self.transcoder_name = transcoder_name
        backend = 'transformerlens'  # change to 'nnsight' for the nnsight backend!
        self.model = ReplacementModel.from_pretrained(
            self.model_name, self.transcoder_name, dtype=torch.bfloat16, backend=backend
        )

    def prune(self):
        if self.graph:
            prune_graph(self.graph, node_threshold=0.7, edge_threshold=0.95)
