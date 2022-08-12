import numpy as np

class LinearRegressor:
    def __init__(self, input_size, transformation=lambda inputs, weights: inputs.dot(weights), print_error=False):
        #last weight is the bias
        self.transformation = transformation
        self.print_error = print_error
        self.weights = np.random.rand(input_size+1)
        self.num_training_inputs = 1

    def __str__(self):
        print(f'Weights: {list(self.weights)[:-1]}')
        print(f'Bias: {list(self.weights)[-1]}')
        ret='y= '
        for i in range(len(self.weights)-1):
            if self.weights[i]<0:
                ret+=f'- {abs(self.weights[i])}*x{i+1} '

            elif self.weights[i]>0 and i==0:
                ret += f'{self.weights[i]}*x{i + 1} '

            else:
                ret += f'+ {self.weights[i]}*x{i + 1} '

        if self.weights[-1]<0:
            ret += f'- {abs(self.weights[-1])}'
        else:
            ret += f'+ {self.weights[-1]}'

        return ret

    # just for training
    def train_predict(self, inputs):
        return self.transformation(inputs, self.weights)

    # just to show error
    def total_mean_squared_error(self, pred_vals, true_vals):
        return sum(((pred_vals-true_vals)**2))/len(true_vals)

    def error_gradient(self, input_i, true_val):
        pred_val=self.train_predict(input_i)
        gradient = (2/self.num_training_inputs)*(pred_val-true_val)*input_i
        return gradient

    def train(self, training_inputs, training_outputs, learning_rate, repeats):
        self.num_training_inputs=len(training_inputs)
        for i in range(len(training_inputs)):
            training_inputs[i].append(1)
            training_inputs[i]=np.array(training_inputs[i])

        for i in range(repeats):
            for j in range(len(training_inputs)):
                gradient=self.error_gradient(training_inputs[j],training_outputs[j])
                self.weights=self.weights-gradient*learning_rate

            # just to show error
            if self.print_error:
                if (i+1)%10==0:
                    pred_vals=[self.train_predict(training_inputs[j]) for j in range(len(training_inputs))]
                    error=self.total_mean_squared_error(np.array(pred_vals),np.array(training_outputs))
                    print(f'Repetition {i+1}; Error: {error}, Weights: {self.weights}')

    def predict(self, inputs):
        inputs.append(1)
        inputs = np.array(inputs)
        return self.transformation(inputs, self.weights)

# testing
class Tester:
    def __init__(self, true_weights, num_data=20, transformation=lambda inputs, weights: inputs.dot(weights)):
        self.weights = true_weights
        self.transformation = transformation
        data = self.make_data(num_data)
        self.data_x=data[0]
        self.data_y=data[1]

    def make_data(self, num_data):
        data_x=[]
        data_y=[]
        for i in range(num_data):
            data_x_i = (10*np.random.rand(len(self.weights)))
            # data_y_i = data_x_i.dot(self.weights)+2*np.random.rand()
            data_y_i=self.transformation(data_x_i, self.weights)
            data_x.append(list(data_x_i))
            data_y.append(data_y_i)
        return data_x, data_y

## linear dataset
test1=Tester([1], 50)
lin1=LinearRegressor(len(test1.data_x[0]))
lin1.train(test1.data_x, test1.data_y, 0.005, 100)
print(lin1)
print('-'*20)
## multivariable linear dataset
test2=Tester([1,2,3], 50)
lin2=LinearRegressor(len(test2.data_x[0]))
lin2.train(test2.data_x, test2.data_y, 0.005, 100)
print(lin2)
print('-'*20)
## quadratic dataset
trans=lambda inputs, weights: weights[0]*inputs[0]**2
trans2=lambda inputs, weights: weights[0]*inputs[0]**2+inputs[1]*weights[1]
test3=Tester([2], 50 ,trans)
lin3=LinearRegressor(len(test3.data_x[0]), trans2)
lin3.train(test3.data_x, test3.data_y, 0.005, 100)
print(lin3)