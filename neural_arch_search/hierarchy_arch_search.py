import torch
import torch.nn as nn

#primitive operations for level 1
OPS = {
    "conv3x3": lambda C: nn.Conv2d(C, C, 3, padding=1),
    "conv5x5": lambda C: nn.Conv2d(C, C, 5, padding=2),
    "maxpool": lambda C: nn.MaxPool2d(3, stride=1, padding=1),
    "identity": lambda C: nn.Identity(),
}

# Level 2: Cell is a directed acyclic graph of primitive operations
class Cell(nn.Module):
    def __init__(self, C, genotype):
        super().__init__()
        self.indices = []
        self.ops = nn.ModuleList()
        for op_name, input_idx in genotype:
            self.ops.append(OPS[op_name](C))
            self.indices.append(input_idx)
    def forward(self, states):
        new_states = []
        for op, idx in zip(self.ops, self.indices):
            # Validate index to provide a clearer error message if it's out of range
            if idx < 0 or idx >= len(states):
                raise IndexError(f"Cell received input index {idx} but only {len(states)} state(s) are available. "
                                 f"Check the genotype {self.indices} and how the cell is placed within its Block.")
            out = op(states[idx])
            new_states.append(out)
        return torch.sum(torch.stack(new_states), dim=0)

#Level 3: Block is a sequence of cells, where each cell takes all previous cell outputs as input
class Block(nn.Module):
    def __init__(self, C, block_structure):
        super().__init__()
        self.cells = nn.ModuleList()
        for genotype in block_structure:
            cell = Cell(C, genotype)
            self.cells.append(cell)
    def forward(self, x):
        states = [x]
        for cell in self.cells:
            x = cell(states)
            states.append(x)
        return x

#Level 4: Network is a sequence of blocks
class HierarchicalNetwork(nn.Module):
    def __init__(self, C, num_blocks, block_structures):
        super().__init__()
        self.stem = nn.Conv2d(3, C, 3, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = Block(C, block_structures[i])
            self.blocks.append(block)
        self.classifier = nn.Linear(C, 10)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean([2, 3])
        return self.classifier(x)

# Level 2: Cell genotype (op, input_state_index)
cell_1 = [('conv3x3', 0), ('identity', 0)]
cell_2 = [('maxpool', 0), ('conv5x5', 1)]
# Level 3: Two blocks, each with a few cells
# Ensure genotypes used as cells reference valid previous states within each block.
# Place cells that expect an index 1 after at least one preceding cell in the same block.
block_structures = [
    [cell_1, cell_2],       # block 1
    [cell_1, cell_2, cell_1]  # block 2 (reordered so indices are valid)
]
model = HierarchicalNetwork(C=32, num_blocks=2, block_structures=block_structures)
# Test forward pass
x = torch.randn(1, 3, 32, 32)
# y = model(x)
print(f"\nmodel: {model}")
y = model.forward(x)
print(y.shape)
