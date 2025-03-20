import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize RNN parameters and hyperparameters.
        
        Args:
            input_size (int): Size of input vector (vocabulary size)
            hidden_size (int): Size of hidden state vector
            output_size (int): Size of output vector (vocabulary size)
            learning_rate (float): Learning rate for gradient descent
        """
        # Initialize weights with small random values to break symmetry
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wz = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases to zero
        self.bz = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        
    def forward(self, inputs, z_prev):
        """
        Forward pass through the RNN.
        
        Args:
            inputs: List of input vectors, each of shape (input_size, 1)
            z_prev: Initial hidden state of shape (hidden_size, 1)
            
        Returns:
            tuple: (output vector, final hidden state)
            
        Notes:
            - Store necessary values in self.* for use in backward pass
            - Use tanh activation for hidden state
            - Use softmax activation for output
        """
        inputs = np.array(inputs)

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)

        if z_prev.ndim == 1:
            z_prev = z_prev.reshape(-1,1)


        h_t = np.tanh(np.dot(self.Wx , inputs) + np.dot(self.Wz , z_prev) + self.bz)

        y_t = self.softmax(np.dot(self.Wy, h_t) + self.by)

        # store for backward
        self.last_input = inputs
        self.last_hidden = h_t
        self.last_prev_hidden = z_prev
        self.last_output = y_t

        return y_t, h_t
        # Your implementation here
        pass
    
    def backward(self, target):
        """
        Backward pass through time (BPTT).
        
        Args:
            target: One-hot encoded target vector
            
        Notes:
            - Implement backpropagation through time
            - Calculate gradients for Wx, Wz, Wy, bz, and by
            - Use stored values from forward pass
            - Update weights and biases using self.learning_rate
        """
        # Your implementation here
        error = (self.last_output - target).reshape(-1,1)

        print("Error shape", error.shape)
        print("self.Wy.T shapeL", self.Wy.T.shape)

        update_Wy = np.dot(error, self.last_hidden.T)
        update_by = error


        change_in_h_t = np.dot(self.Wy.T, error).reshape(-1,1) * (1 - self.last_hidden ** 2)
        
        
        update_Wx = np.dot(change_in_h_t, self.last_input.T)
        update_Wz = np.dot(change_in_h_t, self.last_prev_hidden.T)
        update_bz = change_in_h_t


        self.Wy -= self.learning_rate * update_Wy
        self.by -= self.learning_rate * update_by
        self.Wx -= self.learning_rate * update_Wx
        self.Wz -= self.learning_rate * update_Wz
        self.bz -= self.learning_rate * update_bz

        pass
    
    @staticmethod
    def softmax(x):
        """
        Compute softmax values for vector x.
        
        Args:
            x: Input vector
            
        Returns:
            Vector of same shape as x with softmax probabilities

        Consider numerical stability in your implementation
        """
        # Your implementaton here
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x
    pass
    
    def sample(self, z, seed_char, char_to_idx, idx_to_char, length=100, temperature=1.0):
        """
        Generate text starting with seed_char.
        
        Args:
            z: Initial hidden state
            seed_char: First character to start generation
            char_to_idx: Dictionary mapping characters to indices
            idx_to_char: Dictionary mapping indices to characters
            length: Number of characters to generate
            temperature: Controls randomness (lower = more conservative)
            
        Returns:
            str: Generated text of specified length
        """
        X = np.zeros((len(char_to_idx), 1))
        X[char_to_idx[seed_char]] = 1
        generated = seed_char
        
        for _ in range(length):
            # Forward pass with single character
            o, z = self.forward([X], z)
            
            # Apply temperature scaling
            logits = np.log(o)
            exp_logits = np.exp(logits / temperature)
            probs = exp_logits / np.sum(exp_logits)
            
            # Sample next character from probability distribution
            idx = np.random.choice(len(probs), p=probs.ravel())
            next_char = idx_to_char[idx]
            
            generated += next_char
            X = np.zeros((len(char_to_idx), 1))
            X[char_to_idx[next_char]] = 1
        
        return generated

def create_char_mappings(text):
    """
    Create character-to-index and index-to-character mappings.
    
    Args:
        text: Input text
        
    Returns:
        tuple: (unique characters, char_to_idx dict, idx_to_char dict)
    """
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

def create_training_sequences(text, char_to_idx, seq_length=25):
    """
    Create training sequences and targets from input text.
    
    Args:
        text: Input text
        char_to_idx: Dictionary mapping characters to indices
        seq_length: Length of sequences to generate
        
    Returns:
        tuple: (X training sequences, y target characters)
    """
    sequences = []
    next_chars = []
    
    # Create sequences and their target next characters
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])
    
    # Convert to one-hot encoded vectors
    X = np.zeros((len(sequences), seq_length, len(char_to_idx)))
    y = np.zeros((len(sequences), len(char_to_idx)))
    
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_chars[i]]] = 1
    
    return X, y

def train_rnn(rnn, X, y, epochs=50, batch_size=32, print_every=10):
    """
    Train the RNN model.
    
    Args:
        rnn: SimpleRNN instance
        X: Training sequences
        y: Target characters
        epochs: Number of training epochs
        batch_size: Size of mini-batches
        print_every: How often to print progress and generate samples
    """
    n_samples = len(X)
    
    for epoch in range(epochs):
        total_loss = 0
        indices = np.random.permutation(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            z = np.zeros((rnn.hidden_size, 1))
            
            batch_loss = 0
            for idx in batch_indices:
                inputs = X[idx]
                target = y[idx].reshape(-1,1)

                z = np.zeros((rnn.hidden_size, 1))
                # resets hidden state for each sequence 
                # fixes value error with X

                for t in range(inputs.shape[0]): # seq_length
                    input_t = inputs[t].reshape(-1,1) # shape is (input_size, 1)
                    o, z = rnn.forward(input_t, z)    # process one time step at a time 


                loss = -np.sum(target * np.log(o + 1e-10))
                batch_loss += loss
                
                # Backward pass
                rnn.backward(target)
            
            total_loss += batch_loss
        
        if epoch % print_every == 0:
            avg_loss = total_loss / n_samples
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            # Generate sample text
            z = np.zeros((rnn.hidden_size, 1))
            sample_text = rnn.sample(z, 'T', char_to_idx, idx_to_char, length=100)
            print(f'Sample:\n{sample_text}\n')

def load_data(filename):
    """
    Load text data from file with error handling.
    
    Args:
        filename: Path to text file
        
    Returns:
        str: Content of the file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file can't be decoded as UTF-8
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text:
            raise ValueError("File is empty")
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {filename}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(f"Could not decode file: {filename}. Please ensure it's a valid text file.")

# Test the data loading and processing pipeline
try:
    # Load the sonnets
    text = load_data('sonnets.txt')
    print(f"Successfully loaded {len(text)} characters")
    
    # Create character mappings
    chars, char_to_idx, idx_to_char = create_char_mappings(text)
    print(f"Vocabulary size: {len(chars)} unique characters")
    
    # Create training sequences
    X, y = create_training_sequences(text, char_to_idx, seq_length=25)
    print(f"Created {len(X)} training sequences")
    
    # Print a sample sequence and its target
    sample_idx = 0
    sample_sequence = ''.join([idx_to_char[np.argmax(x)] for x in X[sample_idx]])
    sample_target = idx_to_char[np.argmax(y[sample_idx])]
    print(f"\nSample sequence: {sample_sequence}")
    print(f"Target character: {sample_target}")
    
except Exception as e:
    print(f"Error processing data: {str(e)}")


# importing the text file 

text = load_data("sonnets.txt")
"""
print(text[:500])

"""

chars, char_to_idx, idx_to_chr = create_char_mappings(text)

print(f"Vocabulary size: {len(chars)} unique chars")

X, y = create_training_sequences(text,char_to_idx, seq_length=25)

print(f"Created {len(X)} training sequences")

#intialize rnn
input_size = len(chars)
hidden_size = 25
output_size = len(chars)
learning_rate = 0.01

rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate)

train_rnn(rnn, X, y, epochs=50, batch_size=32, print_every=10)

#generate Text
z = np.zeros((rnn.hidden_size, 1))
seed_char = "S" #text file starts with "Shall I compare thee"

generated_text = rnn.sample(z, seed_char, char_to_idx, idx_to_char, length=200)
print("Generated Text: \n", generated_text)