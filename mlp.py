import jax.numpy as jnp 
import jax
from typing import Tuple 
import matplotlib.pyplot as plt
import random
from functools import partial
from jax.nn import tanh
import optax 

def build_dataset(dataset): 
    
    chars = sorted(list(set(''.join(dataset))))
    stoi  = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos  = {i:s for s,i in stoi.items()} 
    

    block_size = 3
    X, Y = [], []

    for w in dataset: 
        context  = [0] * block_size

        for ch in w + '.': 

            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = jnp.array(X)
    Y = jnp.array(Y)
    return X, Y, itos, stoi

@jax.jit
def linear_layer(C, X):
    #C[X] embedding space

    emb = C[X]

    return emb

@jax.jit
def batchNorm(h: jnp.array): 
    mean = jnp.mean(h, axis=0)
    var = jnp.var(h, axis=0) 
    h = (h - mean) / jnp.sqrt(var + 1e-5)
    return h

@jax.jit
def dropout(h: jnp.array, key: jnp.array, rate: float = 0.8):
    mask = jax.random.bernoulli(key, rate, h.shape)
    return h * mask / rate


@jax.jit
def hidden_layer(h, W, b): 
    
    h = h @ W + b 
    h = tanh(h)
    return h 

@jax.jit
def forward(params: Tuple, X: jnp.array):  

    C, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = params 

    emb = linear_layer(C, X)
    h = emb.reshape(X.shape[0], -1) 
    
    h = hidden_layer(h, W1, b1)
    h = hidden_layer(h, W2, b2)
    h = hidden_layer(h, W3, b3)
    h = hidden_layer(h, W4, b4)
    h = hidden_layer(h, W5, b5)
  

    logits = h @ W6 + b6
    return logits

@jax.jit
def loss_function(logits, Y): 

    loss = -jnp.sum(jax.nn.log_softmax(logits, axis=-1) * jax.nn.one_hot(Y, logits.shape[-1]), axis=-1).mean()
    return loss 

@jax.jit
def compute_loss(params: Tuple, X:jnp.array, Y:jnp.array): 
    logits = forward(params, X)
    return loss_function(logits, Y)

def get_batch(key, X, Y, batch_size): 

    dataset_size = len(X)
    indices = jax.random.randint(key, (batch_size,), 0, dataset_size)
    return X[indices], Y[indices] 

@jax.jit
def init_params(key, n_embd=10, n_hidden=100, vocab_size=27):
    keys = jax.random.split(key, 13)
    
    # Initialize embeddings
    C = jax.random.normal(keys[0], (vocab_size, n_embd))
    
    # Initialize hidden layers
    scale = jnp.sqrt(2.0 / 30)  # for first layer
    W1 = jax.random.normal(keys[1], (30, n_hidden)) * scale
    b1 = jnp.zeros((n_hidden,))
    
    # Rest of hidden layers
    scale = jnp.sqrt(2.0 / n_hidden)
    W2 = jax.random.normal(keys[2], (n_hidden, n_hidden)) * scale
    b2 = jnp.zeros((n_hidden,))
    
    W3 = jax.random.normal(keys[3], (n_hidden, n_hidden)) * scale
    b3 = jnp.zeros((n_hidden,))
    
    W4 = jax.random.normal(keys[4], (n_hidden, n_hidden)) * scale
    b4 = jnp.zeros((n_hidden,))
    
    W5 = jax.random.normal(keys[5], (n_hidden, n_hidden)) * scale
    b5 = jnp.zeros((n_hidden,))
    
    # Output layer with smaller weights
    W6 = jax.random.normal(keys[6], (n_hidden, vocab_size)) * 0.1
    b6 = jnp.zeros((vocab_size,))
    
    return (C, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6)


def train(params: Tuple, X: jnp.array, Y:jnp.array, max_steps: int, batch_size: int):

    scheduler = optax.piecewise_constant_schedule(
        init_value = 0.005, 
        boundaries_and_scales={100000: 0.008}
    )
    optimizer = optax.chain(
        optax.clip(1.0),
        optax.adam(learning_rate=scheduler)
    )

    opt_state = optimizer.init(params)

    @jax.jit
    def training_step(params, opt_state, batch_X, batch_Y):
        loss_val, grads = jax.value_and_grad(compute_loss)(params, batch_X, batch_Y)

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss_val
  

    key = jax.random.PRNGKey(0)

    
    
    for step in range(max_steps):
        
        key, batch_key = jax.random.split(key)
        batch_X, batch_Y = get_batch(batch_key, X, Y, batch_size)
        params, opt_state, loss_val = training_step(params, opt_state, batch_X, batch_Y)

        
        if step % 1000== 0: 
            print(f'{step:7d}/{max_steps:7d}: {loss_val:.4f}')

    return params
  

def sample_from_model(params, itos, block_size=3, num_samples=20):
    key = jax.random.PRNGKey(2147483647 + 10)
    
    C, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = params

    for _ in range(num_samples):
        out = []
        context = [0] * block_size # initialize with padding
        

        for _ in range(20):
            # Get embeddings
            emb = linear_layer(C, jnp.array([context])) # (1,block_size,d)
            
            # Forward pass through first layer
            h = emb.reshape(1, -1) @ W1 + b1
            h = tanh(h)
            
            # Forward through remaining layers
            h = hidden_layer(h, W2, b2)
            h = hidden_layer(h, W3, b3)
            h = hidden_layer(h, W4, b4)
            h = hidden_layer(h, W5, b5)
            
            # Output layer
            logits = h @ W6 + b6
            
            # Get probabilities
            probs = jax.nn.softmax(logits[0])
            
            # Sample from distribution
            key, subkey = jax.random.split(key)
            ix = jax.random.categorical(subkey, probs)
            ix = int(ix)
            
            # Update context
            context = context[1:] + [ix]
            out.append(ix)
            
            # Break if end token or max length reached
            if ix == 0:
                break
        
        print(''.join(itos[i] for i in out))


if __name__ == "__main__": 
    
    df = open('names.txt', 'r').read().splitlines()
    X, Y, itos, stoi= build_dataset(df)

    print(X.shape, Y.shape)
 
    random.seed(42)
    random.shuffle(df)
    n1 = int(0.8*len(df))
    n2 = int(0.8*len(df))

    Xtr,  Ytr,_, _  = build_dataset(df[:n1])     # 80%
    Xdev, Ydev,_ , _= build_dataset(df[n1:n2])   # 10%
    Xte,  Yte,_ , _ = build_dataset(df[n2:])     # 10%


    key = jax.random.PRNGKey(0)

    params = init_params(key)

    params = train(params, Xtr, Ytr, 200000, 32)
    print("###########  Generating sample names: ###################")
    sample_from_model(params, itos)