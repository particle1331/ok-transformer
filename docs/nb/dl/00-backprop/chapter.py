import math
import random
random.seed(42)

from typing import final
from collections import OrderedDict


class Node:
    def __init__(self, data, parents=()):
        self.data = data
        self.grad = 0               # ∂loss / ∂self
        self._parents = parents     # parent -> self

    @final
    def sorted_nodes(self):
        """Return topologically sorted nodes with self as root."""
        topo = OrderedDict()

        def dfs(node):
            if node not in topo:
                for parent in node._parents:
                    dfs(parent)

                topo[node] = None

        dfs(self)
        return reversed(topo)


    @final
    def backward(self):
        """Send global grads backward to parent nodes."""
        self.grad = 1.0
        for node in self.sorted_nodes():
            for parent in node._parents:
                parent.grad += node.grad * node._local_grad(parent)


    def _local_grad(self, parent) -> float:
        """Calculate local grads ∂self / ∂parent."""
        raise NotImplementedError("Base node has no parents.")


    def __add__(self, node):
        return BinaryOpNode(self, node, op="+")

    def __mul__(self, node):
        return BinaryOpNode(self, node, op="*")

    def __pow__(self, n):
        assert isinstance(n, (int, float)) and n != 1
        return PowOp(self, n)

    def relu(self):
        return ReLUNode(self)

    def tanh(self):
        return TanhNode(self)

    def __neg__(self):
        return self * Node(-1)

    def __sub__(self, node):
        return self + (-node)


class BinaryOpNode(Node):
    def __init__(self, x, y, op: str):
        """Binary operation between two nodes."""
        ops = {"+": lambda x, y: x + y, "*": lambda x, y: x * y}
        self._op = op
        super().__init__(ops[op](x.data, y.data), (x, y))

    def _local_grad(self, parent):
        if self._op == "+":
            return 1.0

        elif self._op == "*":
            i = self._parents.index(parent)
            coparent = self._parents[1 - i]
            return coparent.data

    def __repr__(self):
        return self._op


class ReLUNode(Node):
    def __init__(self, x):
        data = x.data * int(x.data > 0.0)
        super().__init__(data, (x,))

    def _local_grad(self, parent):
        return float(parent.data > 0)

    def __repr__(self):
        return "relu"


class TanhNode(Node):
    def __init__(self, x):
        data = math.tanh(x.data)
        super().__init__(data, (x,))

    def _local_grad(self, parent):
        return 1 - self.data**2

    def __repr__(self):
        return "tanh"


class PowOp(Node):
    def __init__(self, x, n):
        self.n = n
        data = x.data**self.n
        super().__init__(data, (x,))

    def _local_grad(self, parent):
        return self.n * parent.data ** (self.n - 1)

    def __repr__(self):
        return f"** {self.n}"


from graphviz import Digraph


def trace(root):
    """Builds a set of all nodes and edges in a graph."""
    # https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb

    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v._parents:
                edges.add((parent, v))
                build(parent)

    build(root)
    return nodes, edges


def draw_graph(root):
    """Build diagram of computational graph."""

    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right
    nodes, edges = trace(root)
    for n in nodes:
        # Add node to graph
        uid = str(id(n))
        dot.node(name=uid, label=f"data={n.data:.3f} | grad={n.grad:.4f}", shape="record")

        # Connect node to op node if operation
        # e.g. if (5) = (2) + (3), then draw (5) as (+) -> (5).
        if len(n._parents) > 0:
            dot.node(name=uid + str(n), label=str(n))
            dot.edge(uid + str(n), uid)

    for child, v in edges:
        # Connect child to the op node of v
        dot.edge(str(id(child)), str(id(v)) + str(v))

    return dot

from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self):
        self._parameters = []

    @final
    def parameters(self) -> list:
        return self._parameters

    @abstractmethod
    def __call__(self, x: list):
        pass

    @final
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, n_in, activation=None):
        self.n_in = n_in
        self.act = activation

        self.w = [Node(random.random()) for _ in range(n_in)]
        self.b = Node(0.0)
        self._parameters = self.w + [self.b]

    def __call__(self, x: list):
        assert len(x) == self.n_in
        out = sum((x[j] * self.w[j] for j in range(self.n_in)), start=self.b)
        if self.act is not None:
            if self.act == "tanh":
                out = out.tanh()
            elif self.act == "relu":
                out = out.relu()
            else:
                raise NotImplementedError("Activation not supported.")
        return out

    def __repr__(self):
        return f"{self.act if self.act is not None else 'linear'}({len(self.w)})"


class Layer(Module):
    def __init__(self, n_in, n_out, *args):
        self.neurons = [Neuron(n_in, *args) for _ in range(n_out)]
        self._parameters = [p for n in self.neurons for p in n.parameters()]

    def __call__(self, x: list):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer[{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, n_in, n_outs, activation=None):
        sizes = [n_in] + n_outs
        self.layers = []
        for i in range(len(n_outs)):
            act = activation if i < len(n_outs) - 1 else None
            layer = Layer(sizes[i], sizes[i + 1], act)
            self.layers.append(layer)

        self._parameters = [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP[{', '.join(str(layer) for layer in self.layers)}]"

