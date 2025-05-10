from models import annotated_tensor
import numpy

class MultiLayerPerceptron:
    layer_sizes: list[int]
    weights: list[annotated_tensor.AnnotatedTensor]

    def __init__(self, layer_sizes: list[int], initial_weights: list[numpy.ndarray] | None = None):
        self.layer_sizes = [*layer_sizes]
        self.weights = []

        if initial_weights == None:
            initial_weights = []
            for i in range(len(layer_sizes) - 1):
                initial_weights.append(numpy.random.normal(
                    loc = 0, scale = 1, size = (layer_sizes[i + 1], layer_sizes[i] + 1)))

        assert len(initial_weights) == len(layer_sizes) - 1, \
            (f"Expected {len(layer_sizes) - 1} weight tensors for {len(layer_sizes)} layers. "
             f"Recieved {len(initial_weights)} weights.")

        for layer_num, weight_tensor in enumerate(initial_weights):
            if weight_tensor.shape != (layer_sizes[layer_num + 1], layer_sizes[layer_num] + 1):
                raise ValueError(f"This is bad TODO")
            self.weights.append(annotated_tensor.AnnotatedTensor(weight_tensor))
    

    def predict(self, features: numpy.ndarray) -> annotated_tensor.AnnotatedTensor:
        assert len(features.shape) == 2
        assert features.shape[1] == self.layer_sizes[0], \
            f"{features.shape[1]} != {self.layer_sizes[0]}"
        
        features = features.transpose()

        input = annotated_tensor.AnnotatedTensor(features)
        for layer_index, weight in enumerate(self.weights):
            input_size = self.layer_sizes[layer_index]
            output_size = self.layer_sizes[layer_index + 1]
            expander = annotated_tensor.AnnotatedTensor(numpy.eye(input_size + 1, input_size, -1))
            input = (expander @ input)
            bias = numpy.zeros_like(input.value)
            bias[..., 0, :] = 1
            input = input + annotated_tensor.AnnotatedTensor(bias)
            input = weight @ input
            input = annotated_tensor.sigmoid(input)
        return input
    

    def reg_cost(self) -> annotated_tensor.AnnotatedTensor:
        current_cost = annotated_tensor.AnnotatedTensor(numpy.array([[0]], float))
        for weight in self.weights:
            n_inputs = weight.value.shape[1]
            weight = weight @ annotated_tensor.AnnotatedTensor(numpy.eye(n_inputs, n_inputs - 1, -1))
            current_cost = current_cost + annotated_tensor.sum(weight * weight * 0.25)
        return current_cost
    

    def get_batch_cost(self, features: list[numpy.ndarray], labels: list[numpy.ndarray], la: float):
        if len(features) != len(labels):
            raise ValueError("Lengths do not match!")
        total_cost = self.reg_cost() * la
        N = len(features)
        for i in range(N):
            prediction = self.predict(features[i])
            label = annotated_tensor.AnnotatedTensor(labels[i])
            current_cost = annotated_tensor.sum(annotated_tensor.cross_entropy(prediction, label))
            total_cost = total_cost + (current_cost * (1 / N))
        return total_cost


    def zero_grad(self):
        for weight in self.weights:
            weight.zero_grad()

    
    def step(self, amount: float):
        for weight_index in range(len(self.weights)):
            self.weights[weight_index] = annotated_tensor.AnnotatedTensor(
                self.weights[weight_index].value + self.weights[weight_index].gradient * amount)


if __name__ == "__main__":
    weights = [numpy.array([[0.40000, 0.10000],
                            [0.30000, 0.20000]]),
               numpy.array([[0.70000, 0.50000, 0.60000]])]
    model = MultiLayerPerceptron([1, 2, 1], weights)

    prediction_one = model.predict(numpy.array([[0.13000]]))
    label_one = annotated_tensor.AnnotatedTensor(numpy.array([[0.9]]))
    print(prediction_one.value)
    print(annotated_tensor.cross_entropy(prediction_one, label_one).value)
