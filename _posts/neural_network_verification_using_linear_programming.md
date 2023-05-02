# Introduction

*Robustness* is a desirable property in a neural network. Informally,
robustness can be described as 'resilience to perturbations in the
input'. Said differently, a neural network is robust if small changes to
the input produce small or no changes to the output. In particular, if
the network is a classifier, robustness means that inputs close to each
other should be assigned the same class by the network. Let us look at
an example.\
Suppose our network assigns class $0$ (blue) to a reference input (thick
blue circle). A robust network would assign the same class (i.e. blue)
to all input points "close\" to the reference point. In particular, a
robust network would not assign the class $1$ (red) to any input close
to the reference point. In the figure below, case I shows a network that
is robust at the reference point while case II shows a network that is
not robust at the reference point.

![image](images/motivating example.png){width=".75\\textwidth"}

Ensuring robustness of networks is important because neural networks are
vulnerable to adversarial examples produced by small perturbations in
the input. E.g. small changes in the image of a chihuahua can lead a
network to classify it as a chocolate chip muffin.
https://www.freecodecamp.org/news/chihuahua-or-muffin-my-search-for-the-best-computer-vision-api-cbda4d6b425d/\
In the subsequent sections, we will make the notion of robustness more
precise. We will then explore how we can verify the robustness of a
*trained* neural network using a very popular idea from mathematical
optimization, viz. *Linear Programming*.

# Problem setup

Suppose we are given a *K-class classifier* fully connected, feed
forward neural network that was trained using the ReLU activation
function. Note that the weights and biases of this network are fixed.
Suppose the network accepts real vectors of length $n$ as inputs. Then
we can represent the network as a classification function $F$ given by:
$\mathbb{R}^n \longrightarrow \{1, 2, ..., K\}$: $$F(x_0) = l_0$$ where
$l_0$ is the class label assigned to the input $x_0$ by the network.

## $F$ as a composition of functions

Let $W_i$ and $b_i$ represent the weight matrix and the bias vector of
the $i$th layer respectively. Then we can write the output of the $i$th
layer recursively as follows: $$\label{eq_nn_output_recursive}
\begin{split}
    z_{i} &= \phi(W_{i} z_{i-1} + b_{i})\\
    &= \phi \circ \omega_i(z_{i - 1})
\end{split}$$

where $\omega_i$ represents the affine transformation corresponding to
the $i$th layer and $\phi$ represents the \"vectorized\" version of the
activation function i.e.
$$\phi(x_1, ..., x_n) = (\phi(x_1), ..., \phi(x_n))$$

We can then describe the network output function
$f_W: \mathbb{R}^n \longrightarrow \mathbb{R}^K$ as follows:
$$\label{eq_output_func}
    f_W(x) = \phi \circ \omega_{H+1} \circ \dots \circ \phi \circ \omega_1(x)$$

where $H$ is the number of hidden layers in the network. Now we can
define a labeling function
$\pi: \mathbb{R}^K \longrightarrow \{1, 2, ..., K\}$ as follows:
$$\label{eq_pi}
\pi(y) = \mathop{\mathrm{arg\,max}}_{i = 1, ... , K} y_i$$

where $K$ is the number of classes. The labeling function $\pi$ selects
the index of the component with the largest value in the output vector
$y$. For a given input $x_0$ to our classifier network, we have:
$$\label{eq_l0}
    F(x_0) = \pi(f_W(x_0) = l_0$$

with $l_0 \in \{1, 2, ..., K\}$. It is worth noting that we can use a
different labeling function here, e.g. *softmax*, without affecting our
analysis as long as the labeling function of choice is non-decreasing.\
![image](images/pi_fw.png){width=".65\\textwidth"}

## Partitioning the co-domain of $f_W$

Suppose we have $K$ class labels. Then our network must output a real
vector of length $K$. Hence, the co-domain of the output function is
fixed to be $\mathbb{R}^K$, regardless of what the actual output
function $f_W$ is.\
Now, consider the set $\mathcal{H}_{l > i}$ of all the points in
$\mathbb{R}^K$ where the $l$th component is greater than the $i$th
component. $$\label{eq_half_plane}
    \mathcal{H}_{l > i} := \{y \in \mathbb{R}^K : y_l > y_i\}$$

Note that $\mathcal{H}_{l > i}$ is a *half-space*. Suppose we fix $l$
and take the intersection of all half-spaces $\mathcal{H}_{l > i}$,
$i \ne l$. We get the set $S_l$ of points where the $l$th component is
greater than every other component. $$\label{eq_S_l}
\begin{split}
    S_l :=& \bigcap_{i \ne l} \mathcal{H}_{l > i}\\
         =& \{y \in \mathbb{R}^K: y_l > y_i, i \ne l\}
\end{split}$$ The set $S_l$ is called the *polyhedron induced by the
class label $l$*. For the sake of simplicity, we will assume that there
are no ties among the components of $y$, i.e.
$i \ne j \implies y_i \ne y_j$. Note that the set of points where
$y_i = y_j$, for some $i$ and $j$, is a zero Lebesgue measure set and
doesn't have a substantial effect on our analysis. In fact, the above
assumption can be relaxed with very little modification to our argument.
Now we make a few observations:

-   $S_l$ is an intersection of half-spaces and is therefore a
    *polyhedron*.

-   $S_i \bigcap S_j = \emptyset$ for $i \ne j$.

-   $\bigcup_{i = 1}^K S_i$ fills up (almost) all of $\mathbb{R}^K$
    (only leaving out a zero Lebesgue measure set which can be ignored
    for the sake of simplicity)

In other words, the co-domain of the neural network output function can
be partitioned into polyhedra induced by the class labels. Now we come
the the main result of this section.

::: proposition
**Proposition 1** (*The image of an input lies inside an induced
polyhedron*). *Let $x_0 \in \mathbb{R}^n$ be an input to a classifier
network with classification function $F$, output function $f_W$ and the
labeling function $\pi$. Then, $$\label{eq_l0_means_image_inside_Sl0}
     F(x) = l \iff f_W(x) \subset S_{l}$$*
:::

::: proof
*Proof.* Let $y := f_W(x)$. Then $l = \pi(y)$. By definition of $\pi$
(see [\[eq_pi\]](#eq_pi){reference-type="ref" reference="eq_pi"}), $l$
is the index of the largest component of $y$. Also, from the definition
of $S_{l}$ (see [\[eq_S\_l\]](#eq_S_l){reference-type="ref"
reference="eq_S_l"}), we know that $S_{l}$ is the set of all points
where the largest component is $l$. Hence we conclude that
$y \in S_{l} \implies f_W(x) \in S_{l}$. The proof in the opposite
direction uses the same argument. ◻
:::

![image](images/pi_maps_Si_to_i.png){width="100%"}

## Formalizing robustness for classifier networks

Recall that we say a classifier network is robust if small perturbations
to an input do not affect its classification. Let us define a set that
contains all 'small' perturbations of the reference input $x_0$. We will
call this set the *region of interest*.

::: definition
**Definition 1**. The set $\chi \subset \mathbb{R}^n$ given by
$$\chi := \mathcal{B}_{\infty}(x_0, \epsilon) = \{x: \parallel x - x_0 \parallel_{\infty} \le \epsilon \}$$
is called the *region of interest*.
:::

For our network to be robust, it must assign the same class to all
points in $\chi$ i.e. $F(x) = F(x_0) = l_0$ for all $x \in \chi$, which
is equivalent to saying $f_W(x) \in S_{l_0}$ for all $x \in \chi$ (see
[\[eq_l0_means_image_inside_Sl0\]](#eq_l0_means_image_inside_Sl0){reference-type="ref"
reference="eq_l0_means_image_inside_Sl0"}). In other words, we want the
image of the region of interest, $\chi$, to lie inside $S_{l_0}$, the
polyhedron induced by $l_0$. The verification problem then reduces to
asking the below set membership question:
$$f_W(\chi) \stackrel{?}{\subset} S_{l_0}$$

We are now ready to give a formal definition of robustness.

::: definition
**Definition 2** ($\epsilon$-robustness). We say that a neural network
is $\epsilon$-robust at $x_0$ if and only if
$$f_W(\mathcal{B}_\infty (x_0, \epsilon)) \subset S_{l_0}$$
:::

::: definition
**Definition 3** (Adversarial example). A point
$\Tilde{x} \in \mathbb{R}^n$ is said to be an *adversarial example* if
$$\Tilde{x} \in \mathcal{B}_\infty (x_0, \epsilon),\ \ f_W(\Tilde{x}) \notin S_{l_0}$$
:::

The two-class classifier network shown in the figure below is not
$\epsilon$-robust at $x_0$ since $\Tilde{x}$ is an adversarial example.

![image](images/adversarial_example.png){width="100%"}

## Verification as an optimization problem

Recall that, given a trained $H$-hidden-layer neural network with the
output function $f_W$, a reference input point $x_0$ and a positive
$\epsilon$, we want to answer the following question:
$$f_W(\mathcal{B}_\infty(x_0, \epsilon)) \stackrel{?}{\subset} S_{l_0}$$
Or, equivalently, $$\label{eq_exists_adv_ex}
    \stackrel{?}{\exists} \Tilde{x} \in \mathcal{B}_\infty(x_0, \epsilon)\ s.\ t.\ f_W(\Tilde{x}) \notin S_{l_0}$$

Before moving forward, we introduce some notation for convenience. Let
$\Tilde{z_i}$ denote the pre- and $z_i$ denote the post-activation
output of the $i$th layer of the network. In other words

$$\label{eq_z_i_z_i_hat}
    \begin{split}
        \Tilde{z_i} &= W_i z_{i -1} + b_i ,\qquad i = 1, ..., H + 1\\
        z_i &= ReLU(\Tilde{z_i}), \qquad i = 1, ..., H + 1
    \end{split}$$ In particular, note that $z_{H+1} = f_W(z_0)$. Using
the above notation,
([\[eq_exists_adv_ex\]](#eq_exists_adv_ex){reference-type="ref"
reference="eq_exists_adv_ex"}) can be posed as a satisfiability problem
in optimization. $$\label{eq_opt1}
\begin{align}
    &\text{Find}\ z_0\\
    \text{s.t.}\ &z_0 \in \mathcal{B}_\infty(x_0, \epsilon) \label{eq_region_const}\\
    &z_{H+1} = f_W(z_0) \label{eq_network_const}\\
    &z_{H+1} \notin S_{l_0} \label{eq_safety_set_const}
\end{align}$$ where $z_0 \in \mathbb{R}^n$ and
$z_{H+1} \in \mathbb{R}^K$ are the decision variables. Note that if
([\[eq_opt1\]](#eq_opt1){reference-type="ref" reference="eq_opt1"}) is
feasible then our network is not robust. Conversely, if
([\[eq_opt1\]](#eq_opt1){reference-type="ref" reference="eq_opt1"}) is
infeasible then our network is robust. As it turns out,
*([\[eq_opt1\]](#eq_opt1){reference-type="ref" reference="eq_opt1"}) is
not a convex optimization problem*. This is because, while the *region
of interest constraint*
([\[eq_region_const\]](#eq_region_const){reference-type="ref"
reference="eq_region_const"}) is convex, the *network constraint*
([\[eq_network_const\]](#eq_network_const){reference-type="ref"
reference="eq_network_const"}) and the *safety set constraint*
([\[eq_safety_set_const\]](#eq_safety_set_const){reference-type="ref"
reference="eq_safety_set_const"}) are not convex. Below, we replace
these non-convex constraints with their convex approximations.\
**Convexifying the network constraint.** Note that
([\[eq_network_const\]](#eq_network_const){reference-type="ref"
reference="eq_network_const"}) is not a convex constraint because $f_W$
is not a convex function. However, we know that $f_W$ is *piece-wise
affine*. Suppose we can find a convex set $\chi' \subset \mathbb{R}^n$,
such that $f_W$ is affine on $\chi'$. Then, we can replace
[\[eq_network_const\]](#eq_network_const){reference-type="ref"
reference="eq_network_const"} with the following convex approximation:
$$\label{eq_network_const_aff}
    z_{H+1} = f_W|_{\chi'}(z_0)$$ where $f_W|_{\chi'}$ is the
restriction of $f_W$ to $\chi'$. But how do we even begin to look for
such a $\chi'$? As a starting point, it is helpful to note that we want
$\chi' \bigcap \mathcal{B}_\infty(x_0, \epsilon)$ to be non-empty. This
is to ensure that
[\[eq_network_const_aff\]](#eq_network_const_aff){reference-type="ref"
reference="eq_network_const_aff"} above does not conflict with
[\[eq_region_const\]](#eq_region_const){reference-type="ref"
reference="eq_region_const"}. Said differently, we want $\chi'$ to
contain $x_0$ as well as points that are "close" to $x_0$. We will use
this idea to find the desired $\chi'$.\
Observe that as our reference input $x_0$ propagates through the
network, it causes some neurons to be "activated" in each layer while
others remain inactive. For a given input this activation pattern is
fixed. Now, the main idea is that *points that are close to $x_0$ are
likely to produce the same activation pattern as $x_0$*. So, it might be
useful to look for a set that contains all inputs that produce the same
activation pattern as $x_0$. Using the notation introduced in
([\[eq_z\_i_z\_i_hat\]](#eq_z_i_z_i_hat){reference-type="ref"
reference="eq_z_i_z_i_hat"}), the pre- and post-activation outputs of
the $i$th layer produced by our reference input $x_0$ are
$$\begin{split}
        \Tilde{x}_i &= W_i {x}_{i-1} + b_i\\
        {x}_i &= ReLU(\Tilde{x}_i)
    \end{split}$$ We say that the $j$th neuron in the $i$th layer is
*activated* by the reference input $x_0$ if the $j$th component of $x_i$
is positive, i.e. $(x_i)_j > 0$. Also, note that
$$(x_i)_j > 0 \iff (\Tilde{x}_i)_j > 0$$ The activation status of the
$j$th neuron in the $i$th layer can then be described by a binary
constant $$\delta_{i,j} =
    \begin{cases}
        1 \quad\text{if}\ (\Tilde{x}_i)_j > 0\\
        0 \quad\text{if}\ (\Tilde{x}_i)_j \le 0
    \end{cases}$$ The activation pattern of the $i$th layer can then be
expressed succinctly by the following $d_i \times d_i$ diagonal matrix
$$\Delta_i :=
    \begin{pmatrix}
        \delta_{i,1} & 0 & 0 & 0 & 0\\
        0 & \delta_{i,2} & 0 & 0 & 0\\
        \vdots & \vdots & \vdots & \vdots & \vdots\\
        0 & 0 & 0 & 0 & \delta_{i,d_i}\\
    \end{pmatrix}$$ where $d_i$ is the number of neurons in the $i$th
layer. Note that the linear operator
$\Delta_i: \mathbb{R}^{d_i} \longrightarrow \mathbb{R}^{d_i}$ is nothing
but a projection map.\
Now, recall the definition of the network output function $f_W$ given in
([\[eq_output_func\]](#eq_output_func){reference-type="ref"
reference="eq_output_func"}) where $\phi$ is chosen to be $ReLU$.
Suppose we were to replace the composition $\phi \circ \omega_i$ with
the composition $\Delta_i \circ \omega_i$ in
([\[eq_output_func\]](#eq_output_func){reference-type="ref"
reference="eq_output_func"}). The resulting output function, $f_{W_0}$,
is given by $$\label{eq_f_W_0}
    f_{W_0} = \Delta_{H+1} \circ \omega_{H+1} \circ \dots \circ \Delta_1 \circ \omega_1$$
We make some important observations about $f_{W_0}$.

1.  $f_{W_0}$ is an affine function

2.  $f_{W_0}(z) = f_W(z)$ for every $z$ that produces the same
    activation pattern as $x_0$.

The first point follows from the fact that $f_{W_0}$ is a composition of
affine functions. The second point follows from the fact that, in
computing the network output, $f_{W_0}$ only considers neurons that were
activated by the reference input $x_0$ and ignores all other neurons (do
you see why?). From the above observations, is seems that a good
candidate for $\chi'$ may be: $$\begin{split}
     \chi' &= \{z \in \mathbb{R}^n : z\ \text{produces the same activation pattern as } x_0 \}\\
     &= \{z \in \mathbb{R}^n: f_W(z) = f_{W_0}(z)\}
\end{split}$$ So, finding $\chi'$ simply reduces to solving the equation
$$\begin{split}
    f_W(z) &= f_{W_0}(z)\\
    \phi \circ \omega_{H+1} \circ \dots \circ \phi \circ \omega_1(z) &= \Delta_{H+1} \circ \omega_{H+1} \circ \dots \circ \Delta_1 \circ \omega_1(z)
\end{split}$$ which can be written as
$$\phi \circ \omega_i (z_{i-1}) = \Delta_i \circ \omega_i (z_{i - 1})\ ,\ i = 1, ..., H+1$$
which is equivalent to solving for $z_0$ in
$$\label{eq_ReLU_equals_Delta}
    ReLU(W_i z_{i-1} + b_i) = \Delta_i (W_i z_{i-1} + b_i)\ ,\ i = 1, ..., H+1$$
Solving
([\[eq_ReLU_equals_Delta\]](#eq_ReLU_equals_Delta){reference-type="ref"
reference="eq_ReLU_equals_Delta"}) directly is hard. Fortunately,
([\[eq_ReLU_equals_Delta\]](#eq_ReLU_equals_Delta){reference-type="ref"
reference="eq_ReLU_equals_Delta"}) has the below equivalent affine
formulation. $$\label{eq_affine_equiv}
     (2 \Delta_i - I)(W_i z_{i -1} + b_i) \ge 0\ ,\ i = 1, ..., H+1$$ It
can be shown that every $z_0, z_1, ..., z_{H+1}$ that is a solution to
([\[eq_ReLU_equals_Delta\]](#eq_ReLU_equals_Delta){reference-type="ref"
reference="eq_ReLU_equals_Delta"}) is also a solution to
([\[eq_affine_equiv\]](#eq_affine_equiv){reference-type="ref"
reference="eq_affine_equiv"}) and vice-versa (see proof in appendix).
Moreover, for any $z$ that satisfies
([\[eq_affine_equiv\]](#eq_affine_equiv){reference-type="ref"
reference="eq_affine_equiv"}), we have
$$ReLU(\Tilde{z}_i) = \Delta_i \Tilde{z}_i$$
