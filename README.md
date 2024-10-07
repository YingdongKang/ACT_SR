# ACT-SR: Aggregation Connection Transformer for Remote Sensing Image Super-Resolution
Official Pytorch implementation of the paper "[ACT-SR: Aggregation Connection Transformer for Remote Sensing Image Super-Resolution]

Recently, Transformer-based methods have shown impressive performances in remote sensing image super resolution (RSISR). However, the application of Transformer in RSISR still leads to artifacts and the loss of image detail, due to the monotonous way of integrating information and limitation of unidimensional self-attention. To solve the above problems, an Aggregation Connection Transformer (ACT-SR) is proposed for RSISR, where an effective attention mechanism is used to further enrich information aggregation and enlarge the receptive fields of Transformer model. Specifically, it contains a new aggregation connection Transformer block (ACTB) to capture spatial similarity and channel importance, and these information were aggregated together through series and parallel connection to activate larger receptive field to extract features more efficiently from local to global and across spatial to channel dimensions. Furthermore, a new gated feed-forward network (GFN) was introduced to better provide a nonlinear mapping to the ACTB and control the information flow. In addition, ACT-SR adopts shift windows scheme and residual learning scheme to efficiently recover detail and eliminate artifacts. In the experiments, the effectiveness of the proposed modules was verified, and the proposed ACT-SR demonstrated superior performance over several state-of-the-art RSISR methods.

## Requirements
PyTorch >= 1.7
BasicSR == 1.4.2


## Installation
Clone or download this code and install aforementioned requirements 
```
cd codes
```


