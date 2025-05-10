import numpy

class AnnotatedTensor:
    _value: numpy.ndarray
    _gradient: numpy.ndarray

    def __init__(self, value: numpy.ndarray):
        self._value = value.astype(float)
        self._gradient = numpy.zeros_like(self._value)


    def add_gradient(self, amount: numpy.ndarray):
        if amount.shape != self._gradient.shape:
            raise ValueError(
                f"Expected shape {self._gradient.shape}, received shape {amount.shape}")
        self._gradient += amount


    def zero_grad(self):
        self._gradient = numpy.zeros_like(self._gradient)


    @property
    def value(self):
        return self._value


    @property
    def gradient(self):
        return self._gradient
    

    def __add__(self, rhs: "AnnotatedTensor") -> "Sum":
        return Sum(self, rhs)


    def __sub__(self, rhs: "AnnotatedTensor") -> "AnnotatedTensor":
        return self + (-rhs)

    def __matmul__(self, rhs: "AnnotatedTensor") -> "MatrixProduct":
        return MatrixProduct(self, rhs)

    
    def __neg__(self) -> "AnnotatedTensor":
        return self * -1


    def __mul__(self, rhs: "AnnotatedTensor | float") -> "AnnotatedTensor":
        if isinstance(rhs, AnnotatedTensor):
            return Product(self, rhs)
        else:
            scale = AnnotatedTensor(numpy.full_like(self.value, rhs, float))
            return self * scale


class MatrixProduct(AnnotatedTensor):
    lhs: AnnotatedTensor
    rhs: AnnotatedTensor

    def __init__(self, lhs: AnnotatedTensor, rhs: AnnotatedTensor):
        if len(lhs.value.shape) != 2:
            raise ValueError(f"Received matrix with shape {lhs.value.shape}. Broadcasted matmul is not yet supported.")
        if len(rhs.value.shape) < 2:
            raise ValueError(f"Received matrix with shape {lhs.value.shape}. Vectors must be represented as 1 wide matrices.")
        if lhs.value.shape[1] != rhs.value.shape[-2]:
            raise ValueError(f"Received tensors with shape {lhs.value.shape} and {rhs.value.shape}. Dimensions do not match.")
            
        super().__init__(lhs.value @ rhs.value)
        self.lhs = lhs
        self.rhs = rhs
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.rhs.add_gradient(self.lhs.value.swapaxes(-1, -2) @ amount)
        self.lhs.add_gradient(amount @ self.rhs.value.swapaxes(-1, -2))


class Product(AnnotatedTensor):
    lhs: AnnotatedTensor
    rhs: AnnotatedTensor

    def __init__(self, lhs: AnnotatedTensor, rhs: AnnotatedTensor):
        if lhs.value.shape != rhs.value.shape:
            raise ValueError("Shapes do not match")
            
        super().__init__(lhs.value * rhs.value)
        self.lhs = lhs
        self.rhs = rhs
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.rhs.add_gradient(amount * self.lhs.value)
        self.lhs.add_gradient(amount * self.rhs.value)


class Sum(AnnotatedTensor):
    lhs: AnnotatedTensor
    rhs: AnnotatedTensor

    def __init__(self, lhs: AnnotatedTensor, rhs: AnnotatedTensor):
        if lhs.value.shape != rhs.value.shape:
            raise ValueError(f"Received tensors with shape {lhs.value.shape} and {rhs.value.shape}. Dimensions do not match.")

        super().__init__(lhs.value + rhs.value)
        self.lhs = lhs
        self.rhs = rhs
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.rhs.add_gradient(amount)
        self.lhs.add_gradient(amount)


class Total(AnnotatedTensor):
    input: AnnotatedTensor

    def __init__(self, input: AnnotatedTensor):
        super().__init__(numpy.array([[numpy.sum(input.value)]]))
        self.input = input
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.input.add_gradient(numpy.full_like(self.input.value, amount.item(), float))


def sum(x: AnnotatedTensor) -> AnnotatedTensor:
    return Total(x)


class ReLUResult(AnnotatedTensor):
    input: AnnotatedTensor

    def __init__(self, input: AnnotatedTensor):
        super().__init__(input.value * (input.value > 0))
        self.input = input
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.input.add_gradient(amount * (self.input.value > 0))
    

def relu(x: AnnotatedTensor) -> AnnotatedTensor:
    return ReLUResult(x)


class SigmoidResult(AnnotatedTensor):
    input: AnnotatedTensor

    def __init__(self, input: AnnotatedTensor):
        value = (input.value < 0) - 2 * ((input.value < 0) - 0.5) * numpy.reciprocal(1 + numpy.exp(-numpy.abs(input.value)))
        super().__init__(value)
        self.input = input
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.input.add_gradient(amount * (self.value) * (1 - self.value))


def sigmoid(x: AnnotatedTensor) -> AnnotatedTensor:
    return SigmoidResult(x)


class NatLogResult(AnnotatedTensor):
    input: AnnotatedTensor

    def __init__(self, input: AnnotatedTensor):
        super().__init__(numpy.log(input.value))
        self.input = input
    

    def add_gradient(self, amount):
        super().add_gradient(amount)
        self.input.add_gradient(amount / self.input.value)


def log(x: AnnotatedTensor) -> AnnotatedTensor:
    return NatLogResult(x)


def cross_entropy(prediction: AnnotatedTensor, label: AnnotatedTensor | numpy.ndarray):
    if not isinstance(label, AnnotatedTensor):
        label = AnnotatedTensor(label)

    if prediction.value.shape != label.value.shape:
        raise ValueError(f"Shapes do not match: {prediction.value.shape} {label.value.shape}")
    
    one = AnnotatedTensor(numpy.ones_like(prediction.value, float))

    return -label * log(prediction) - (one - label) * log(one - prediction)