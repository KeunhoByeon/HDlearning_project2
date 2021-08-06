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

### Test Result
#### Test 01 - OnlineHD with non-linear encoder, Normalize (epoch 20)
```
Loading...
Trainset    size: (6238, 617)    data range: -0.076225 ~ 0.076225
Testset     size: (6238, 617)    data range: -0.073082 ~ 0.073082
Training...
Validating...
acc = 0.980282
acc_test = 0.935215
t = 0.790847
```

#### Test 02 - OnlineHD with non-linear encoder, Normalize (epoch 40)
```
Loading...
Trainset    size: (6238, 617)    data range: -0.076225 ~ 0.076225
Testset     size: (6238, 617)    data range: -0.073082 ~ 0.073082
Training...
Validating...
acc = 0.994550
acc_test = 0.937139
t = 1.490470
```

#### Test 03 - OnlineHD with Linear encoder, no Normalize (epoch 40) (Encoding tooks about 5m)
```
Trainset    size: (6238, 617)    data range: -1.000000 ~ 1.000000
Testset     size: (1559, 617)    data range: -1.000000 ~ 1.000000
Encoding...
Training...
Validating...
acc = 0.993267
acc_test = 0.927518
t = 2.502428
```