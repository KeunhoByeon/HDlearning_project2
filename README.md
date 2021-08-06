# OnlineHD - Linear encoder
Original code: https://gitlab.com/biaslab/onlinehd

## How to run

### Non-linear Encoder
```
python3 example.py
```
### Linear Encoder
```
python3 example_linear.py
```

### Note
Added Linear Encoder at ./onlinehd/encoder.py line 89

I changed the way to set 'flipped D/(M−1) bits' to make Lm hypervector
```python
# ./onlinehd/encoder.py line 100
# I changed this method
flip = int(self.dim / i)
# to this to flip entire vector linearly
flip = int(self.dim / m) * i
```
