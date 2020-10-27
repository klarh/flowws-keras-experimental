
def sequence(x, layers):
    for layer in layers:
        x = layer(x)
    return x
