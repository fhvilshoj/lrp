# LRP Framework

This project aims to implement the LRP rules for any [tensorflow](https://www.tensorflow.org):1.4 graph consisting of simple components such as linear layers, convolutions, LSTMs and simple pooling layers. See *Status* to see how far we have come.

# Status
### (alpha, beta)-rule, epsilon-rule, W^2-rule and flat-rule
- [x] Linear layers
- [x] Convolutions
- [x] Max pooling
- [x] Nonlinearities
- [x] LSTM
- [x] Concatenates
- [x] Splits
- [x] Tile
- [x] Reshaping (Reshape, Expand_dims, Squeeze)
- [x] Sparse matrix multiplication
- [x] Sparse reshape

# Usage
A simple usage of the framework:
```python
from lrp import lrp

with g.as_default():
    inp = ...
    pred = ...
    config = LRPConfiguration()
    # Set propagation rule for, e.g., linear layers
    config.set(LAYER.LINEAR, AlphaBetaConfiguration(alpha=2, beta=-1))

    # Calculate the relevance scores using lrp
    expl = lrp.lrp(inp, pred, config)

    with tf.Session() as sess:
        # Compute prediction and explanation
        prediction, explanation = sess.run([pred, expl], feed_dict={inp: ...})
```

