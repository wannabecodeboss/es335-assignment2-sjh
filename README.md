**Comments about Task 1:**
>Convergence Speed: Momentum accelerated convergence for both datasets in BGD and SGD, reaching the ϵ-neighborhood faster than vanilla GD.
>For Dataset 1, both BGD and SGD with momentum converged much quicker due to consistent gradient directions.
>For Dataset 2, BGD with momentum converged in ~147 steps, but SGD with momentum took ~8 × 10⁵ steps because of small feature magnitudes and an overly large learning rate causing oscillations.

>Momentum Behavior: The momentum term averaged recent gradients, smoothing the trajectory and reducing oscillations. Over iterations, the momentum direction aligned with the gradient direction, providing faster, smoother descent.

>Contour Plot Insights: Red arrows (gradients) showed the instantaneous direction, blue arrows (momentum) showed the accumulated direction, and the black line (θ trajectory) became smoother with momentum—showing reduced zig-zag movement compared to vanilla GD.

>Summary: Momentum greatly accelerates convergence when parameters are well-tuned but can slow or destabilize training if gradients are small or noisy.

**Comments about Task 2:**
Without giving any limit, we used a keyboard interrupt to arbitrarily stop the training after some time.

**Comments about Task 3:**
We filtered the input data as per our discretion (averaging the temperature over 10 days to reduce the noise.

**Comments about Task 4:** 
>We observed that the reconstructed image for the rectangular patch using Gradient Decent gave us better results compared to ALS method. While both gave good results where 900 random pixels where missing.

>The general trend observed was that as rank r increased from 5 to 50, the RMSE reduced and PSNR increased. 

>The patch with just one colour had the least RMSE and highest PSNR while the patch with 3 or more colours had least PSNR and higher RMSE

**Comments about Task 5:** 
Increased the iterations from 1000 to 10000 for better convergence.

The .py files have been reuploaded as .ipynb files for better reading. Do not consider them as separate submissions.

