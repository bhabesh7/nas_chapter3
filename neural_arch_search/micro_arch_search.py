# Micro Search Space Example: Cell-based design (like NASNet or DARTS)
import torch
import torch.nn as nn

OPS = {
    'conv3x3': lambda C_in, C_out: nn.Conv2d(C_in, C_out, 3, padding=1),
    'maxpool': lambda C_in, C_out: nn.MaxPool2d(3, stride=1, padding=1),
    'skip': lambda C_in, C_out: nn.Identity(),
}

class Cell(nn.Module):
    def __init__(self, C_in, C_out, genotype):
        super().__init__()
        # Keep modules in a ModuleList so their parameters are registered
        # Store input indices separately.
        self.ops = nn.ModuleList()
        self.input_indices = []
        for op_name, input_idx in genotype:
            op = OPS[op_name](C_in, C_out)
            self.ops.append(op)
            self.input_indices.append(input_idx)

    def forward(self, inputs):
        states = [inputs]
        for i, (input_idx, op) in enumerate(zip(self.input_indices, self.ops)):
            new_state = op(states[input_idx])
            states.append(new_state)
        return torch.sum(torch.stack(states[1:], dim=0), dim=0)
    
# Example genotype describing the cell connections
genotype = [
    ('conv3x3', 0),
    ('maxpool', 1),
    ('skip', 0),
]

# Repeat cell to form a full model
class MicroNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, padding=1)
        self.cells = nn.ModuleList([Cell(32, 32, genotype) for _ in range(3)])
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.stem(x)
        for cell in self.cells:
            x = cell(x)
        x = x.mean([2, 3])
        return self.fc(x)
   
model = MicroNetwork()
print(f"\nmodel: {model}")

forward = model.forward(torch.randn(1, 3, 32, 32))  # Test forward pass with dummy input
print(f"forward.shape: {forward.shape} \n")
