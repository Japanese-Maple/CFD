# CFD

$$
\begin{cases}
\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} = 0 \\
\dfrac{\partial u}{\partial t} + u \dfrac{\partial u}{\partial x} + v \dfrac{\partial u}{\partial y} = -\dfrac{1}{\rho} \dfrac{\partial p}{\partial x} + \nu \left( \dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2} \right) + f_x \\
\dfrac{\partial v}{\partial t} + u \dfrac{\partial v}{\partial x} + v \dfrac{\partial v}{\partial y} = -\dfrac{1}{\rho} \dfrac{\partial p}{\partial y} + \nu \left( \dfrac{\partial^2 v}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2} \right) + f_y
\end{cases}
$$

$$
\begin{cases}
\nabla \cdot \mathbf{u} = 0 \\
\dfrac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\dfrac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}
\end{cases}
$$
