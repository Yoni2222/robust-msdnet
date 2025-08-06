This repository includes a PyTorch-based implementation of a multi-exit convolutional neural network with confidence‑gated early exits, 
trained using PGD adversarial training enhanced with Adversarial Weight Perturbation (AWP). 
Designed to strike a balance between inference efficiency and adversarial robustness on CIFAR‑10 and CIFAR‑100.

Features
Multi-scale convolutional backbone with multiple classifier heads

Learnable confidence gates enabling conditional early exit per sample

Adversarial robustness via integrated PGD + AWP training

Setup Instructions
Clone the repository

Install dependencies
Prepare a virtual environment and run:

bash
Copy
Edit
pip install torch torchvision numpy matplotlib autoattack
Optionally:

bash
Copy
Edit
pip freeze > requirements.txt
Enable dataset download
Ensure your configuration uses download=True for CIFAR datasets so the data is fetched automatically during first run.

Running Training & Evaluation
Everything can be run via command-line arguments to main_awp.py, using either your system terminal or PyCharm’s integrated terminal (same functionality).

Typical training command:


python main_awp.py --data cifar10 --data-root ./data \
  --arch gated_mixture --nBlocks 5 --batch-size 128 \
  --attack pgd --epsilon 8 --alpha 1 --attack-iters 40 --norm l_inf \
  --awp-gamma 0.005 --awp-warmup 5

Evaluation (after training):
python main_awp.py --eval-only --adv-eval --use-gates-infer --gate-thresh 0.7 \
  --evaluate-from path/to/model_checkpoint.pth.tar

- Results & Figures
Scripts are included to reproduce the following results:

Standard, PGD-40, and AutoAttack robust accuracies

Exit-mix distribution (percent of samples exiting at each threshold τ)

Per-exit robustness when evaluated on all samples (no early exit)

Visualizations available for CIFAR‑10 and CIFAR‑100

- Reproducibility Notes
Tested with Python 3.8+, PyTorch 2.x on CUDA-enabled GPU

Dataset is downloaded to ./data/ during first run

Ensure correct thresholds and n‑blocks match training settings when evaluating