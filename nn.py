import random, math
class Value:
    def __init__(self,data,children=(),op=''):
        self.data=float(data); self.grad=0.0; self._backward=lambda:None; self._prev=set(children)
    def __add__(self,o):
        o=o if isinstance(o,Value) else Value(o)
        out=Value(self.data+o.data,(self,o))
        def _b(): self.grad+=out.grad; o.grad+=out.grad
        out._backward=_b; return out
    def __mul__(self,o):
        o=o if isinstance(o,Value) else Value(o)
        out=Value(self.data*o.data,(self,o))
        def _b(): self.grad+=o.data*out.grad; o.grad+=self.data*out.grad
        out._backward=_b; return out
    def tanh(self):
        t=math.tanh(self.data); out=Value(t,(self,))
        def _b(): self.grad+=(1-t**2)*out.grad
        out._backward=_b; return out
    def backward(self):
        topo=[]; vis=set()
        def build(v):
            if v not in vis: vis.add(v); [build(c) for c in v._prev]; topo.append(v)
        build(self); self.grad=1.0
        for v in reversed(topo): v._backward()
    def __radd__(self,o): return self+o
    def __rmul__(self,o): return self*o
    def __neg__(self): return self*-1
    def __sub__(self,o): return self+(-o)
class Neuron:
    def __init__(self,nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(0)
    def __call__(self,x): return sum((wi*xi for wi,xi in zip(self.w,x)),self.b).tanh()
    def parameters(self): return self.w+[self.b]
class Layer:
    def __init__(self,nin,nout): self.neurons=[Neuron(nin) for _ in range(nout)]
    def __call__(self,x): return [n(x) for n in self.neurons]
    def parameters(self): return [p for n in self.neurons for p in n.parameters()]
class MLP:
    def __init__(self,nin,nouts):
        sz=[nin]+nouts; self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers: x=layer(x)
        return x[0] if len(x)==1 else x
    def parameters(self): return [p for l in self.layers for p in l.parameters()]
if __name__=="__main__":
    random.seed(42)
    nn=MLP(3,[4,4,1])
    xs=[[2,3,-1],[3,-1,0.5],[0.5,1,1],[-1,0.5,-1]]
    ys=[1,-1,-1,1]
    for epoch in range(100):
        preds=[nn(x) for x in xs]
        loss=sum((p-y)*(p-y) for p,y in zip(preds,ys))
        for p in nn.parameters(): p.grad=0.0
        loss.backward()
        for p in nn.parameters(): p.data+=-0.05*p.grad
    preds=[nn(x).data for x in xs]
    correct=sum(1 for p,y in zip(preds,ys) if (p>0)==(y>0))
    print(f"MLP: {correct}/4 correct, loss={loss.data:.4f}")
    assert correct>=3
    print("All tests passed!")
