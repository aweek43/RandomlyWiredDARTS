WS(Generate Random DAG):
cd RandomWiredGraph && python3 ws.py -k 4 -p 0.75 -o ws.txt

Model Search
1. Make wired-cell based on generated DAG
2. cd cnn && python3 train_search.py --unrolled

Evaluation Model
1. Make wired-cell based on generated DAG(same as model search)
2. cd cnn/search-EXP-{your model}/scripts && python3 train.py --cutout

Performance(CIFAR-10 Top-1 accuracy)
- Model Search
	Baseline: 88.172%
	Randomly Wired DARTS: 90.144%

- Augmented Retraining
	Baseline: 97.13%
	Randomly Wired DARTS: 97.25%

Reference
Radomly Wired Neural Network: https://github.com/seungwonpark/RandWireNN
DARTS: https://github.com/quark0/darts