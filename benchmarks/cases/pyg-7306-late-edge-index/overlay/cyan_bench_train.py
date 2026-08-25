import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

GOOD_GRAPHS = 80
FINAL_EDGE = 3


class GraphClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = GCNConv(4, 4)
        self.output = torch.nn.Linear(4, 2)

    def forward(self, graph: Data) -> torch.Tensor:
        hidden = self.conv(graph.x, graph.edge_index).relu()
        return self.output(hidden.mean(dim=0, keepdim=True))


def graph(edge: int) -> Data:
    return Data(
        x=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ),
        edge_index=torch.tensor([[0, 1, 2, edge], [1, 2, 3, 0]], dtype=torch.long),
        y=torch.tensor([1]),
    )


torch.manual_seed(17)
model = GraphClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
graphs = [graph(3) for _ in range(GOOD_GRAPHS)] + [graph(FINAL_EDGE)]
for index, sample in enumerate(graphs, start=1):
    print(
        f"training-graph={index} nodes={sample.x.size(0)} max_edge={int(sample.edge_index.max())}",
        flush=True,
    )
    optimizer.zero_grad()
    logits = model(sample)
    loss = torch.nn.functional.cross_entropy(logits, sample.y)
    loss.backward()
    optimizer.step()
print("training-complete", flush=True)
