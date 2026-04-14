# Method

MLeml implements a compact, EML-only symbolic regressor inspired by the gradient-based section of the reference paper.

## Representation

The model is a full binary tree of fixed depth.

- every internal node is the same operator: `eml(left, right)`
- leaves choose from a small terminal set
- for 1D that set is `{1, x}`
- for 2D that set is `{1, x1, x2}`

## Soft training stage

The tree starts as a differentiable circuit.

- leaf terminals are selected by softmax logits
- each internal node has two sigmoid gates
- each gate chooses between the child value and the constant `1`
- the node then applies `EML`

This creates a continuous relaxation of a discrete symbolic tree.

## Optimization

Training is full-batch and deterministic given the internal restart seeds.

- multiple restarts with different initial biases
- `Adam` optimization
- intermediate value clamping to reduce overflow and NaN cascades
- entropy and binarity penalties to encourage discrete choices
- temperature annealing during the hardening phase

## Snapping

After optimization, the soft tree is projected to a discrete tree:

- leaf terminal = `argmax(leaf_logits)`
- gate choice = `sigmoid(gate_logit) >= 0.5`

The package evaluates the snapped candidate and returns the best one across restarts.

## Scope

This implementation is intentionally modest.

- it focuses on one or two input variables
- it is designed for shallow depths
- it favors robustness and readable output over exhaustive symbolic search

