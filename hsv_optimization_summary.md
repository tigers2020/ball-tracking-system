# HSV mask Optimization summation

## outline
this The document CPU Foundation HSV mask Function and GPU Foundation HSV mask Functional Performance Optimization The result Summary. Full HD(1920x1080) Image In use various In conditions Tested.

## test environment
- Used image: 1920x1080 Sized synthesis image (Red, Blue, green, Yellow one include)
- background: Gradation + Noise
- repeat number: 20-50episode

## Performance result summation

### single image treatment Performance
| avatar method     | average treatment hour | speed elevation |
|--------------|-------------|---------|
| CPU avatar      | 3.27 ms     | Reference value   |
| GPU avatar      | 1.50 ms     | 2.18x   |

### arrangement treatment Performance
| arrangement size | entire treatment hour | Image treatment hour | single image Contrast speed elevation |
|----------|-------------|-----------------|-------------------|
| 1        | 1.50 ms     | 1.50 ms         | 1.00x             |
| 2        | 1.75 ms     | 0.88 ms         | 1.72x             |
| 4        | 4.14 ms     | 1.04 ms         | 1.45x             |
| 8        | 9.11 ms     | 1.14 ms         | 1.32x             |
| 16       | 17.38 ms    | 1.09 ms         | 1.38x             |

## Optimization detail

### CPU avatar Optimization
1. color(Hue) treatment Logic improvement
   - color Range Boundaries Passing case(yes: Red 170°-10°) Efficiently treatment
   - two doggy Mask Create bit OR By operations combination

2. Mopologue calculation adjustment
   - Noise Remove For corrosion(erode) calculation apply
   - Connectivity Improve For expansion(dilate) calculation repeat number increase

### GPU avatar Optimization
1. Memory Optimization
   - `torch.no_grad()` Context In use Memory amount used decrease
   - use after middle Tenser Off Memory Efficiency elevation
   - Efficient Tenser Operation For channel extraction and absorption

2. calculation Optimization
   - logic Operation For `torch.logical_and`/`torch.logical_or` use
   - unnecessary Memory copy and conversion Minimization
   - Kernel generation and Reuse Optimization

3. arrangement treatment backup
   - various arrangement size treatment backup
   - arrangement In size Follow -up automatic Chunk By hand Memory Overflow prevention
   - various input Tenser form(channel order, arrangement dimension) automatic treatment

## Performance Insight
1. GPU Implementation CPU Contrast approximately 2.18ship speed
2. arrangement size 2of case Image treatment Time most Enemy (0.88 ms)
3. arrangement treatment city small arrangement(2-4)go Large -scale Batch Efficiency
4. beginning GPU Warm -up Time I need it, since Performance Consistent

## In the future Optimization direction
1. Kernel size automatic adjustment Mechanism Introduction
   - image In size according to Mopologue Computational Kernel size automatic adjustment

2. Memory amount used addition Optimization
   - Buffer Full Used Memory Reuse
   - unnecessary middle Tenser eliminate

3. Hybrid approach avatar
   - image Size and System In a state according to CPU/GPU avatar automatic select
   - small Image Low load In the situation CPU use, One Large -scale arrangement treatment city GPU use

4. model Quantification apply
   - calculation Precision While maintaining Performance Improvement For half-precision apply examine 