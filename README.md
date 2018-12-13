# Mitigating-poisoning-attack-by-Shapley-value
Temporal code repo for "Mitigating poisoning attack by Shapley value"

Hard to read, many debug informations contained.
But it can be executed well and flexible to modify its structure.

A lot of code-clean work to be done.. :(


First execute run.py with parameters eps = 0.2, \#poisoning data = 0.2 * 360 = 240
```
python run.py --dataset "MNIST | CIFAR | ImageNet" --eps 0.2 --num 240 >> resultsnew.txt
```

Then run rubbish.py to collect data from "resultsnew.txt" to generate "poison.pkl"
```
python rubbish.py --num 240
```

Finally run eval.py to evaluate the performance made by Oracle, Shapley value, influence function and random removal:
```
python eval.py --dataset "MNIST | CIFAR | ImageNet" --eps 0.2 --num 240
```
