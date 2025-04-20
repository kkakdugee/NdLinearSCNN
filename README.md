# 🧠 Sparse Coding with NdLinear

This project explores **sparse coding**, a brain-inspired technique where only a small number of neurons activate at a time, just like in the human brain. The goal: build efficient, interpretable representations.

Instead of using `nn.Linear`, this model uses [NdLinear](https://github.com/ensemble-core/NdLinear), a drop-in replacement that reduces parameters and maintains tensor structure.

Beyond benchmarking, I wanted to explore a more biologically grounded approach to neural efficiency, showing how sparsity and structure (via NdLinear) can combine to reduce overhead while retaining performance.

## What’s Inside

- Sparse autoencoders (with `NdLinear` and `nn.Linear`) trained on MNIST 
- KL-divergence sparsity penalty
- Visuals of reconstruction and learned features
- Clear performance vs. efficiency comparison

## 📊 Results

| Metric                  | `nn.Linear` Model| `NdLinear` Model |
|-------------------------|------------------|------------------|
| Param Count             | ~402K            | ~2K              |
| True Sparsity Level (%) | ~96%             | ~54%             |
| Convergence Rate        | 11               | 9                |
| Reconstruction Loss     | 0.012            | 0.014            |
| Sparsity (KL) Loss      | 0.00017          | 0.00016          |
| Total Loss              | 0.017            | 0.025            |
| Avg. Inference Time (s) | 1.62             | 1.64             |
| Avg. Training Time (s)  | 11.24            | 11.06            |

> While NdLinear shows a lower true sparsity, it achieves comparable reconstruction performance with just 0.5% of the parameters used by standard Linear layers, demonstrating a remarkable parameter efficiency and a compelling trade-off between model size and representational properties.


## Key Takeaway

NdLinear represents a significant advancement in neural network efficiency, requiring 200x fewer parameters while maintaining reconstruction quality and preserving meaningful sparse representations, pointing toward more biologically plausible and computationally efficient deep learning architectures.

## Running

To run this project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/NdLinearSCNN.git
   cd NdLinearSCNN
   ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the benchmarks**
    ```python
    python benchmark.py --mode all
    ```

4. **View Results**
    Results are saved in the benchmark_results directory. Each experiment has its own timestamped folder.

5. **Visualize specific metrics**
    ```python
    # To visualize without retraining
    python benchmark.py --mode visualize --checkpoint benchmark_results/experiment_TIMESTAMP
    ```

The code supports both CPU and CUDA. In my case, training 15 epochs with a NVIDIA RTX 3070 GPU took approx. 5-6 minutes. 

To view available commands,
```bash
python benchmark.py -h
```

## 📚 References

- Olshausen & Field (1997), *Sparse coding with an overcomplete basis set* [[link](https://www.sciencedirect.com/science/article/pii/S0042698997001697)]
- Rao & Ballard (1999), *Predictive coding in the visual cortex* [[link](https://www.cs.utexas.edu/~dana/nn.pdf)]
- Ensemble (2024), *NdLinear: Structured Efficiency in Deep Learning* [[arXiv](https://arxiv.org/abs/2503.17353)]
- Andrew Ng, *Sparse Autoencoder Tutorial* [[pdf](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)]
- Goodfellow et al. (2016), *Deep Learning* (MIT Press), Chapter 14 [[book](https://www.deeplearningbook.org/)]