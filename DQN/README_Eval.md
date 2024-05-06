## Eval

The `Eval` class is utilized to evaluate the models performance facilitating batch episode runs and metric output.

### Usage

A standard 10 episode batch evaluation:
```python
# Create an instance of the Eval class
eval = Eval()

# Run the evaluation process
eval.run()
```

Expected output:
```
Time elapsed:   0.32541847229003906
---------------------------------------------
Goal Achieved Count:                    7
Goal Not Achieved Count:                3
Accuracy:                               70.00%
---------------------------------------------
Episode 0. Distance from goal 33.83784863137726. Termination event 1. Initial distance 598.652653882032.
Episode 1. Distance from goal 39.67366884975475. Termination event 1. Initial distance 362.6761640913282.
Episode 2. Distance from goal 51.39066063011839. Termination event 0. Initial distance 766.0815883442181.
Episode 3. Distance from goal 39.08964057138413. Termination event 1. Initial distance 550.5706130915453.
Episode 4. Distance from goal 67.00746227100382. Termination event 0. Initial distance 888.7688113339711.
Episode 5. Distance from goal 35.11409973215888. Termination event 1. Initial distance 615.8839176338346.
Episode 6. Distance from goal 32.863353450309965. Termination event 1. Initial distance 682.0850386865262.
Episode 7. Distance from goal 61.69278726074872. Termination event 0. Initial distance 830.3168070080239.
Episode 8. Distance from goal 35.86084215408221. Termination event 1. Initial distance 613.454154766271.
Episode 9. Distance from goal 32.155870381627054. Termination event 1. Initial distance 543.3175866838842.
Failed missions: average initial distance 60.03030338729031, final distance 828.3890688954043
```

While the test episode `batch_size` is under or equal to 20 the above metrics are expected. Beyond 20, the following output is observed:
```
python eval.py --batch_size 100
Time elapsed:   2.457918882369995
---------------------------------------------
Goal Achieved Count:                    69
Goal Not Achieved Count:                31
Accuracy:                               69.00%
---------------------------------------------
Failed missions: average initial distance 55.269048740505134, final distance 680.4885109285924
```
