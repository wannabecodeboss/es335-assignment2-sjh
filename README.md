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

**Comments about Task 5:** 
Increased the iterations from 1000 to 10000 for better convergence.
