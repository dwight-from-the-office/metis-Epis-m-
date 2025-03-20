import numpy as np

with open("sonnets.txt", "r") as file:

    text = file.read().lower()


words = text.split()
unique_words = sorted(set(words))


word_to_index = {word: i for i, word in enumerate(unique_words)}
index_to_word = {i: word for i, word in enumerate(unique_words)}

vocab_size = len(unique_words)
print(f"Vocab Size: {vocab_size}")


encoded_text = [word_to_index[word] for word in words]

seq_length = 10

X , Y = [], []

for i in range(len(encoded_text) - seq_length):
    X.append(encoded_text[i:i + seq_length])
    Y.append(encoded_text[i + 1:i + seq_length + 1])

X = np.array(X)
Y = np.array(Y)

print(f"X Shape: {X.shape}")

print(f"Y Shape: {Y.shape}")

class RNN_Model:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size)* 0.01 #input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 #hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 # hidden to output


        # bias
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    
    def forward(self, inputs, h_prev):
        seq_length = len(inputs)
        
        h_states = np.zeros((seq_length + 1, self.hidden_size, 1))
        h_states[0] = np.copy(h_prev)
        y_outputs = np.zeros((seq_length, self.output_size, 1))

       

        for t, x_t in enumerate(inputs):
            x_t_encoded = np.zeros((self.input_size, 1))
            x_t_encoded[x_t] = 1


            h_states[t + 1] = np.tanh(
                np.dot(self.Wxh, x_t_encoded) + np.dot(self.Whh, h_states[t]) + self.bh #input to hidden
            )
            
            y_outputs[t] = np.dot(self.Why, h_states[t+1]) + self.by

        return y_outputs, h_states



    def backward(self, inputs, targets, y_outputs, h_states):
        # initialize gradiants with zeros matrix
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        
        # Biases 
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # gradiant of next hidden state
        dh_next = np.zeros_like(h_states[0])


        for t in reversed(range(len(inputs))):

            #softmax
            y_softmax = np.exp(y_outputs[t] - np.max(y_outputs[t])) / np.sum(np.exp(y_outputs[t] - np.max(y_outputs[t])))

            dy = np.copy(y_softmax)
            dy[targets[t]] -= 1


            dWhy += np.dot(dy, h_states[t+1].T)
            dby += dy



            # backpropagate into hidden states
            dh = np.dot(self.Why.T, dy) + dh_next
        #   deriv_relu = np.where(h_states[t+1] > 0, dh, 0)

            
            deriv_tanh = (1 - h_states[t+1] ** 2) * dh
            
            # to reshape matix need to turn the int index to be one hot encoded
            x_t_encoded = np.zeros((self.input_size, 1))
            x_t_encoded[inputs[t]] = 1
            
            # input to hidden weights (Wxy)
            dWxh += np.dot(deriv_tanh, x_t_encoded.T)

            # hidden to hidden weights (Whh)
            dWhh += np.dot(deriv_tanh, h_states[t].T)

            # gradiant for hidden bias
            dbh += deriv_tanh

            dh_next = np.dot(self.Whh.T, deriv_tanh)

        return dWxh, dWhh, dWhy, dbh, dby 

    def update_weights(self, dWxh, dWhh, dWhy, dbh, dby):
        for param in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(param, -5, 5, out=param)

        g_smooth = 0.9 
        self.Wxh = g_smooth * self.Wxh + self.learning_rate * dWxh
        self.Whh = g_smooth * self.Whh + self.learning_rate * dWhh
        self.Why = g_smooth * self.Why + self.learning_rate * dWhy
        self.bh  = g_smooth * self.bh + self.learning_rate * dbh
        self.by  = g_smooth * self.by + self.learning_rate * dby



    
    def train(self, X, Y, seq_length, epochs):
        h_prev = np.zeros((self.hidden_size, 1))

        #hidden state zeros
        
        for epoch in range(epochs):
            epoch_loss = 0


            for i in range(len(X)):
                loss = 0
                inputs = X[i]
                targets = Y[i]
                # forward pass 
                y_outputs, h_states = self.forward(inputs, h_prev)

                # crross entropy
                for t in range(len(inputs)):
                    y_softmax = np.exp(y_outputs[t] - np.max(y_outputs[t])) / np.sum(np.exp(y_outputs[t] - np.max(y_outputs[t])))
                    
                    loss += -np.log(y_softmax[targets[t]] + 1e-8) / len(inputs)
                loss /= len(inputs)
                epoch_loss += loss
                
                
                #backward pass
                dWxh, dWhh, dWhy, dbh, dby = self.backward(inputs, targets, y_outputs, h_states)
                #update weights
                self.update_weights(dWxh, dWhh, dWhy, dbh, dby)
                
                #last hidden state
                h_prev = h_states[-1]
                
            
                
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss/ len(X)}")
        print("Training Complete...")



    def generate_text(self, start_word=None, length=50, temperature=1.0):
        def temperature_sample(probs, T=1.0):
            probs = np.log(probs + 1e-8)/ T
            probs = np.exp(probs) / np.sum(np.exp(probs))
            return np.random.choice(range(len(probs)), p=probs.ravel())

        if start_word is None:
            start_word = np.random.choice(list(word_to_index.keys()))

        generated_text = [start_word]
        h_prev = np.zeros((self.hidden_size, 1))

        word_idx = word_to_index[start_word]
        for _ in range(length):
            
            x_t_encoded = np.zeros((self.input_size, 1))
            x_t_encoded[word_idx] = 1


          # forward pass
            h_prev = np.tanh(np.dot(self.Wxh, x_t_encoded) + np.dot(self.Whh, h_prev) + self.bh) #activation function
          # h_prev = np.maximum(0, np.dot(self.Wxh, x_t_encoded) + np.dot(self.Whh, h_prev) + self.bh)
          
            y_output = np.dot(self.Why, h_prev) + self.by
            
            probs = np.exp(y_output - np.max(y_output)) / np.sum(np.exp(y_output - np.max(y_output)))

            word_idx = temperature_sample(probs, T=temperature)

            generated_text.append(index_to_word[word_idx])

        return " ".join(generated_text)
    


# create RNN model
rnn = RNN_Model(input_size=vocab_size,hidden_size=256,output_size=vocab_size, learning_rate=0.01)


# train model
rnn.train(X, Y, seq_length=25, epochs=100)


# generated new text
print(rnn.generate_text(length=50,temperature=1.2))
