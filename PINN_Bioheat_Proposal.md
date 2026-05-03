# Physics-Informed Neural Networks (PINNs) for Inverse Pennes Bioheat

## 1. Overview and Motivation

The next step in the 3D Breast Thermography reconstruction pipeline is identifying the metabolic heat source (potential tumor) inside the breast based on the surface temperature map derived from the five canonical TIFF views. 

The traditional approach to this inverse thermal problem involves using Finite Element Method (FEM) software (such as COMSOL Multiphysics or ANSYS). The standard FEM inverse workflow consists of:
1. Guessing a tumor location, size, and heat generation rate.
2. Generating a 3D tetrahedral mesh of the solid breast.
3. Solving the forward Pennes Bioheat equation (a massive matrix inversion) to predict surface temperatures.
4. Comparing the prediction to the real surface data to calculate an error.
5. Updating the parameters and repeating steps 1-4 thousands of times.

This iterative FEM approach is heavily computationally expensive and requires complicated meshing pipelines.

**The Solution:** We propose bypassing traditional FEM entirely by using a **Physics-Informed Neural Network (PINN)** natively inside PyTorch.

---

## 2. What is a PINN?

A PINN is a deep learning architecture designed to solve Partial Differential Equations (PDEs). Instead of discretizing the domain into a tetrahedral mesh, a PINN is a "mesh-free" continuous function approximator.

We build a neural network $T_\theta(x, y, z)$ that takes physical 3D coordinates as input and predicts the temperature at that point. Because the network is built in PyTorch, we can use `autograd` to calculate exact, continuous spatial derivatives (like $\nabla^2 T$) analytically, without any mesh finite differences.

We train the network using a composite loss function that enforces both **measured data** and **physical laws**.

---

## 3. The Mathematics of the Bioheat PINN

The steady-state Pennes Bioheat Equation is defined as:

$$ k \nabla^2 T + \omega_b c_b (T_a - T) + Q_m + Q_{tumor}(x, y, z) = 0 $$

Where:
* $k$: Thermal conductivity of breast tissue
* $\omega_b$: Blood perfusion rate
* $c_b$: Specific heat of blood
* $T_a$: Arterial blood temperature
* $Q_m$: Metabolic heat generation of healthy tissue
* $Q_{tumor}$: Localized heat generation from the tumor

### The Loss Function Structure

The PINN minimizes a total loss consisting of two parts: $\mathcal{L}_{total} = \lambda_{data}\mathcal{L}_{data} + \lambda_{PDE}\mathcal{L}_{PDE}$

**1. The Data Loss ($\mathcal{L}_{data}$):**
We sample points $(x_i, y_i, z_i)$ exclusively on the **surface** of the 3D breast geometry. We enforce that the network's predicted temperature matches the absolute `.tiff` thermography data we projected.
$$ \mathcal{L}_{data} = \frac{1}{N_{surf}} \sum_{i=1}^{N_{surf}} \Big| T_\theta(x_i, y_i, z_i) - T_{measured}(x_i, y_i, z_i) \Big|^2 $$

**2. The Physics Loss ($\mathcal{L}_{PDE}$):**
We sample random "collocation points" $(x_j, y_j, z_j)$ anywhere **inside** the 3D volume (where our learned `vol_np == 1`). At these points, we don't know the true temperature, but we *do* know the PDE must equal zero. We use PyTorch `autograd` to compute $\nabla^2 T$.
$$ f(x, y, z) = k \nabla^2 T_\theta + \omega_b c_b (T_a - T_\theta) + Q_m + Q_{tumor}(x, y, z) $$
$$ \mathcal{L}_{PDE} = \frac{1}{N_{vol}} \sum_{j=1}^{N_{vol}} \Big| f(x_j, y_j, z_j) \Big|^2 $$

---

## 4. Solving the Inverse Problem (Finding the Tumor)

The true power of the PINN approach is in solving the inverse problem simultaneously with the forward problem.

We model the tumor heat source $Q_{tumor}$ as a Gaussian distribution or a sphere:
$$ Q_{tumor}(x, y, z) = Q_{max} \cdot \exp\left(-\frac{(x - x_t)^2 + (y - y_t)^2 + (z - z_t)^2}{r_t^2}\right) $$

Instead of hardcoding these tumor parameters, we define them as **learnable PyTorch parameters**:
* `x_t, y_t, z_t` (Tumor location coordinates)
* `r_t` (Tumor radius)
* `Q_{max}` (Tumor heat generation)

During training, the PyTorch optimizer (e.g., Adam or L-BFGS) updates both the neural network weights $\theta$ *and* the tumor parameters simultaneously. As the network figures out how to satisfy the surface data and the internal physics, the tumor parameters organically slide into the correct physical location to make the math work out!

---

## 5. Implementation Advantages

Why is this perfect for the current `3DBreastNet` pipeline?
1. **Mesh-Free:** We don't need to generate a complex volumetric tetrahedral mesh. We simply use the $128 \times 128 \times 128$ binary `vol_np` array as a boolean mask to sample $(x, y, z)$ collocation points.
2. **Unified Stack:** The entire pipeline from U-Net segmentation $\rightarrow$ 3DBreastNet Volume Generation $\rightarrow$ PINN Tumor Discovery stays exactly within Python and PyTorch.
3. **No Commercial Licenses:** You don't need expensive COMSOL or ANSYS licenses, making the research infinitely more reproducible.
4. **Scale:** Because we are using PyTorch, we can scale the $(x, y, z)$ inputs to realistic biological dimensions (e.g., multiplying the bounding box by 200mm) instantly within the dataset loader.
