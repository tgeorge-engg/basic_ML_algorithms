import numpy as np

class LogisticRegressor:
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
        return 1/(1+np.exp(-self.transformation(inputs, self.weights)))

    # just to show error
    def total_BCE(self, pred_vals, true_vals):
        return np.sum((true_vals*np.log(pred_vals) + (1-true_vals)*np.log(1-pred_vals)))*(1/len(pred_vals))

    def error_gradient(self, input_i, true_val):
        pred_val=self.train_predict(input_i)
        gradient = (1/self.num_training_inputs)*(pred_val-true_val)*input_i
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
                    error=self.total_BCE(np.array(pred_vals),np.array(training_outputs))
                    print(f'Repetition {i+1}; Error: {error}, Weights: {self.weights}')

    def predict(self, inputs):
        inputs.append(1)
        inputs = np.array(inputs)
        return 1/(1+np.exp(-self.transformation(inputs, self.weights)))

#testing
x=[[1,1],[2,2],[3,3],[4,4],[5,5]]
y=[0,0,0,1,1]

lgr=LogisticRegressor(len(x[0]), print_error=True)
lgr.train(x,y,0.5,300)
for i in range(1,7):
    print(lgr.predict([i,i]))

