# Pruing and Quantization of Model
## Pruning
 The process of removing redundant or less important connections (weights) and neurons from the model. 
```python
def prune_weights(weights, pruning_percentage):
    """ Prune the weights based on the specified percentage. """
    # Calculate the threshold for pruning
    num_weights = weights.numel()
    threshold_index = int(num_weights * pruning_percentage)

    # Get the absolute values of weights and find the threshold
    abs_weights = weights.abs().view(-1)
    threshold = torch.kthvalue(abs_weights, threshold_index).values.item()

    # Create a mask for pruning
    mask = abs_weights >= threshold
    pruned_weights = weights * mask.view_as(weights)

    return pruned_weights
```
 A simple weight pruning technique, which is a common method to reduce the size and complexity of neural networks.

1. Calculate Threshold:
    - **num_weights:** Calculates the total number of weights in the input tensor.
    - **threshold_index:** Determines the index of the weight that will be the threshold for pruning. This is calculated based on the desired pruning percentage. For example, if pruning_percentage is 20%, then 20% of the weights with the smallest absolute values will be pruned.
    - **abs_weights:** Calculates the absolute values of all weights.
    - **threshold:** Finds the weight value at the threshold_index position. Weights with absolute values less than this threshold will be pruned.

2. Create Pruning Mask:

    - **mask:** Creates a boolean mask where True indicates weights to be kept and False indicates weights to be pruned.

3. Apply Pruning:

    - **pruned_weights:** Multiplies the original weights by the mask, effectively setting the pruned weights to zero.

## Quantization
Quantization is a technique used to reduce the size and computational cost of deep learning models by converting their weights and activations from high-precision floating-point numbers (e.g., FP32) to lower precision formats (e.g., INT8).

```python
def quantize(self, weights):
        # Upcast the weights to FP32 for stability
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)
        self.int8_weights = int8_weights
        self.scales = scales

```
1. Upcast to FP32:

    - `w_fp32 = weights.clone().to(torch.float32)`: Creates a copy of the original weights (weights) and casts them to FP32 format. This ensures stability during calculations, especially when dealing with very small weight values.

2. Calculate Scale Factors:

    - `scales = w_fp32.abs().max(dim=-1).values`: This calculates the maximum absolute value of each weight across its last dimension (usually the channel dimension).
    - `scales /= 127`: The maximum values are then divided by 127 (assuming an INT8 target format). This scaling factor will be used to map the weights to the limited range of INT8 (-128 to 127).
3. Quantize Weights:

    - `int8_weights = torch.round(weights / scales.unsqueeze(1))`: The original weights are divided by the corresponding scaling factor (scales). The unsqueeze(1) operation expands the scales tensor to have the same number of dimensions as the weights, allowing for element-wise division.
    - `to(torch.int8)`: Finally, the result is rounded using torch.round and converted to the INT8 data type, representing the quantized weights.
4. Store Quantization Parameters:

    - This function likely stores the int8_weights (quantized weights) and scales (scaling factors) as internal attributes (self.int8_weights and self.scales) for later use during inference with the quantized model.


### Quantized Layers

![](./img/Stage-1/Quant+prune.gif)

### Memory Footprint after Pruning and Quantization

| LoRA Tiny BERT (MB) | After Pruing and Quantizing (MB) | Percentage Reduction (%) |
|---|---|---|
| 54.74 | 41.49 | 24.21 |

## Evaluation of the model
| Metric | Value |
|---|---|
| Accuracy | 0.47868 |
| Precision | 0.4755200800782364 |
| Recall | 0.47868 |
| F1-score | 0.4612957278396014 |

















# [Federated Learning ->](02_Stage-2.md)
## [<- Fine Tuning ](01-2_Stage-1.md)
