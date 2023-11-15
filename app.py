import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, laplace, cauchy
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd
from IPython.display import HTML
import base64


# Function to get distribution object
def get_distribution(dist_name, mean, std):
    if dist_name == "Normal":
        return norm(loc=mean, scale=std)
    elif dist_name == "Uniform":
        return uniform(loc=mean , scale=std-mean)
    elif dist_name == "Laplace":
        return laplace(loc=mean, scale=std)
    elif dist_name == "Cauchy":
        return cauchy(loc=mean, scale=std)

def get_scale(target_dist, proposal_dist):
    x = np.linspace(min(target_dist.ppf(0.05), proposal_dist.ppf(0.05)), max(target_dist.ppf(0.95), proposal_dist.ppf(0.95)), 1000)
    return max(target_dist.pdf(x) / proposal_dist.pdf(x))

# Rejection Sampling Function
def rejection_sampling(target_dist, proposal_dist, proposal_scale, n_samples):
    samples = []
    accepted_count = 0
    for _ in range(n_samples):
        x_proposal = proposal_dist.rvs()
        u = np.random.uniform(0, 1)
        num = target_dist.pdf(x_proposal)
        den = proposal_scale * proposal_dist.pdf(x_proposal)
        if u <= num / den:
            samples.append(x_proposal)
            accepted_count += 1
    acceptance_ratio = accepted_count / n_samples
    return np.array(samples), acceptance_ratio

# Importance Sampling Function
def importance_sampling(target_dist, proposal_dist, n_samples):
    samples = proposal_dist.rvs(size=n_samples)
    weights = target_dist.pdf(samples) / proposal_dist.pdf(samples)
    return samples, weights

# Gibbs Sampling Function (Simplified version for demonstration)
def gibbs_sampling_normal(initial_state, n_samples, mean, covariance):
    samples = np.zeros((n_samples, 2))
    samples[0] = initial_state

    sigma1 = np.sqrt(covariance[0, 0])
    sigma2 = np.sqrt(covariance[1, 1])
    rho = covariance[0, 1] / (sigma1 * sigma2)

    for i in range(1, n_samples):
        x2 = samples[i - 1, 1]
        conditional_mean_x1 = mean[0] + rho * sigma1 / sigma2 * (x2 - mean[1])
        conditional_var_x1 = (1 - rho**2) * sigma1**2
        samples[i, 0] = norm(conditional_mean_x1, np.sqrt(conditional_var_x1)).rvs()

        x1 = samples[i, 0]
        conditional_mean_x2 = mean[1] + rho * sigma2 / sigma1 * (x1 - mean[0])
        conditional_var_x2 = (1 - rho**2) * sigma2**2
        samples[i, 1] = norm(conditional_mean_x2, np.sqrt(conditional_var_x2)).rvs()

    return samples

# Gibbs Sampling Function (Simplified version for demonstration)
def gibbs_sampling_uniform(initial_state, n_samples, a, b):
    samples = np.zeros((n_samples, 2))
    samples[0] = initial_state

    for i in range(1,n_samples):
        samples[i, 0] = uniform.rvs(loc=a[0], scale=b[0]-a[0])
        samples[i, 1] = uniform.rvs(loc=a[1], scale=b[1]-a[1])

    return samples


# Streamlit App
st.title("Sampling Methods in Statistics with Interactive Visualization")
st.write("This application provides an interactive way to explore and understand different sampling methods used in statistics and machine learning.")
# Rejection Sampling Section
st.header("Rejection Sampling")


import streamlit as st

st.markdown(r'''  
**Mathematical Explanation**      
Rejection sampling is used to generate samples from a distribution $$ p(x) $$ by using a proposal distribution $$ q(x) $$ from which it is easy to sample. The idea is to sample from $$ q(x) $$ and accept the sample with probability $$ \frac{p(x)}{Mq(x)} $$ where $$ M $$ is chosen such that $$ Mq(x) \geq p(x) $$ for all $$ x $$.
- Acceptance Probability: $$ \frac{p(x)}{Mq(x)} $$
- Sample: Accepted if $$ U \leq \frac{p(x)}{Mq(x)} $$ where $$ U \sim \text{Uniform}(0, 1) $$

**Advantages**
Advantages
- Conceptually simple and easy to implement.
- Effective when the proposal distribution is close to the target distribution.

**Disadvantages**
- Can be inefficient if the proposal distribution is not close to the target distribution, leading to a very low acceptance ratio.
- The efficiency drops significantly in higher dimensions.
''')

# Distribution choices
proposal_dist_choices = ["Normal", "Laplace", "Cauchy"]
dist_choices = ["Normal", "Uniform", "Laplace", "Cauchy"]
proposal_dist_name_rs = st.selectbox("Proposal Distribution for Rejection Sampling", proposal_dist_choices)
target_dist_name_rs = st.selectbox("Target Distribution for Rejection Sampling", dist_choices)

#add some space
st.write("")
st.write("Enter the parameters for the proposal and target distributions for Rejection Sampling")
# Distribution parameters
mean_proposal_rs, std_proposal_rs = st.columns(2)
mean_target_rs, std_target_rs = st.columns(2)

if proposal_dist_name_rs != "Uniform":
    mean_proposal_rs_val = mean_proposal_rs.number_input("Mean of Proposal Distribution (RS)", value=0.0, format="%.2f")
    std_proposal_rs_val = std_proposal_rs.number_input("Std of Proposal Distribution (RS)", value=1.0, min_value=0.1, format="%.2f")
else:
    mean_proposal_rs_val = mean_proposal_rs.number_input("Start of Proposal Distribution (RS)", value=0.0, format="%.2f")
    std_proposal_rs_val = std_proposal_rs.number_input("End of Proposal Distribution (RS)", value=1.0, min_value=0.1, format="%.2f")

if target_dist_name_rs != "Uniform":
    mean_target_rs_val = mean_target_rs.number_input("Mean of Target Distribution (RS)", value=1.0, format="%.2f")
    std_target_rs_val = std_target_rs.number_input("Std of Target Distribution (RS)", value=1.0, min_value=0.1, format="%.2f")
else:
    mean_target_rs_val = mean_target_rs.number_input("Start of Target Distribution (RS)", value=0.0, format="%.2f")
    std_target_rs_val = std_target_rs.number_input("End of Target Distribution (RS)", value=1.0, min_value=0.1, format="%.2f")

n_samples_rs = st.slider("Number of samples for Rejection Sampling", 100, 5000, 1000)

if st.button('Run Rejection Sampling'):
    proposal_dist = get_distribution(proposal_dist_name_rs, mean_proposal_rs_val, std_proposal_rs_val)
    target_dist = get_distribution(target_dist_name_rs, mean_target_rs_val, std_target_rs_val)

    calculated_scale = get_scale(target_dist, proposal_dist)
    samples_rs, acceptance_ratio_rs = rejection_sampling(target_dist, proposal_dist, calculated_scale, n_samples_rs)
    st.write(f"Acceptance Ratio (Rejection Sampling): {acceptance_ratio_rs:.2f}")
    
    # Final plot with KDE
    st.write("Animation for Rejection Sampling- might take a while to load!")
    
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='KDE of Accepted Samples')
    scat = ax.scatter([], [], marker='x', color='black', label='Accepted Samples')
    target_line, = ax.plot([], [], label='Target Distribution')
    proposal_line, = ax.plot([], [], label='Proposal Distribution')

    def update(frame):
        new_samples_list=samples_rs[:len(samples_rs)*(frame+1)//100]
        # print(len(new_samples_list))
        x = np.linspace(min(samples_rs) - 1, max(samples_rs) + 1, 1000)
        # x = np.linspace(-10, 10, 1000)
         #set x limit according to mean of target and proposal + 2std or max of samples
        ax.clear()
        sns.kdeplot(new_samples_list, ax=ax, color='blue', label='KDE of Accepted Samples')  # Plot the KDE directly on the axis
        # line.set_data([], [])
        ax.scatter(new_samples_list, np.zeros_like(new_samples_list), marker='x', color='black')
        # scat.set_offsets(np.column_stack((new_samples_list, np.zeros_like(new_samples_list))))
        ax.plot(x, target_dist.pdf(x), label='Target Distribution')
        ax.plot(x, proposal_dist.pdf(x), label='Proposal Distribution')
        ax.plot(x, calculated_scale*proposal_dist.pdf(x), label='Scaled Proposal Distribution')
        ax.legend(loc='upper right')
        y_lim = max(max(target_dist.pdf(x)), calculated_scale*max(proposal_dist.pdf(x)))+0.05
        ax.set_ylim(-0.025, y_lim)
        ax.set_xlabel('$x$')
        ax.set_ylabel('Density')
        ax.set_title(f"Rejection Sampling: {len(new_samples_list)} Accepted Samples; Proposal Scaling Factor: {calculated_scale:.2f}")
        # target_line.set_data(x, target_dist.pdf(x))
        # proposal_line.set_data(x, proposal_dist.pdf(x))
        return line, scat, target_line, proposal_line

    ani = FuncAnimation(fig, update, frames=100, blit=True)
    ani.save('rs.gif', writer='Pillow', fps=10)

    file_ = open("rs.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )
    

# Importance Sampling Section
st.header("Importance Sampling")
st.markdown(r"""
**Mathematical Explanation**
         
Importance sampling is a technique to estimate properties of a distribution, while having samples generated from a different distribution. It involves re-weighting the samples from a proposal distribution $ q(x) $ to match the target distribution $ p(x) $.
- Weight of each sample: $ w(x) = \frac{p(x)}{q(x)} $

**Advantages**
- Efficient as every sample from $ q(x) $ is accepted (could be with a lower weight).
- Useful in scenarios where sampling from $ p(x) $ is challenging.

**Disadvantages**
- If $ q(x) $ is not similar to $ p(x) $, the variance of the weights can be large, leading to unreliable estimates.
- Choosing an effective proposal distribution can be non-trivial.
""")

proposal_dist_name_is = st.selectbox("Proposal Distribution for Importance Sampling", proposal_dist_choices)
target_dist_name_is = st.selectbox("Target Distribution for Importance Sampling", dist_choices)

mean_proposal_is, std_proposal_is = st.columns(2)
mean_target_is, std_target_is = st.columns(2)

st.write("Enter the parameters for the proposal and target distributions for Importance Sampling")

if proposal_dist_name_is != "Uniform":
    mean_proposal_is_val = mean_proposal_is.number_input("Mean of Proposal Distribution (IS)", value=0.0, format="%.2f")
    std_proposal_is_val = std_proposal_is.number_input("Std of Proposal Distribution (IS)", value=1.0, min_value=0.1, format="%.2f")
else:
    mean_proposal_is_val = mean_proposal_is.number_input("Start of Proposal Distribution (IS)", value=0.0, format="%.2f")
    std_proposal_is_val = std_proposal_is.number_input("End of Proposal Distribution (IS)", value=1.0, min_value=0.1, format="%.2f")

if target_dist_name_is != "Uniform":
    mean_target_is_val = mean_target_is.number_input("Mean of Target Distribution (IS)", value=1.0, format="%.2f")
    std_target_is_val = std_target_is.number_input("Std of Target Distribution (IS)", value=1.0, min_value=0.1, format="%.2f")
else:
    mean_target_is_val = mean_target_is.number_input("Start of Target Distribution (IS)", value=0.0, format="%.2f")
    std_target_is_val = std_target_is.number_input("End of Target Distribution (IS)", value=1.0, min_value=0.1, format="%.2f")

n_samples_is = st.slider("Number of samples for Importance Sampling", 100, 5000, 1000)

if st.button('Run Importance Sampling'):
    proposal_dist = get_distribution(proposal_dist_name_is, mean_proposal_is_val, std_proposal_is_val)
    target_dist = get_distribution(target_dist_name_is, mean_target_is_val, std_target_is_val)

    samples_is, weights_is = importance_sampling(target_dist, proposal_dist, n_samples_is)
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='KDE of Accepted Samples')
    scat = ax.scatter([], [], marker='x', color='black', label='Accepted Samples')
    target_line, = ax.plot([], [], label='Target Distribution')
    proposal_line, = ax.plot([], [], label='Proposal Distribution')
    # Plot for Importance Sampling
    st.write("Animation for Importance Sampling- might take a while to load!")

    def update(frame):
        new_samples_list=samples_is[:len(samples_is)*(frame+1)//100]
        new_weights_list=weights_is[:len(weights_is)*(frame+1)//100]
        ax.clear()
        df = pd.DataFrame({'samples': new_samples_list, 'weights': new_weights_list})
        sns.kdeplot(data=df, x='samples', weights='weights', label='Weighted Samples KDE')
        # x = np.linspace(min(samples_is)-1, max(samples_is)+1, 1000)
        x = np.linspace(-10, 10, 1000)
        ax.plot(x, target_dist.pdf(x), label='Target Distribution')
        ax.plot(x, proposal_dist.pdf(x), label='Proposal Distribution')
        #set y limit according to max of pdf of target and proposal
        y_lim = max(max(target_dist.pdf(x)), max(proposal_dist.pdf(x)))+0.05
        ax.set_ylim(-0.025, y_lim)
        ax.set_title(f"Importance Sampling: {len(new_samples_list)} Weighted Samples")
        ax.scatter(new_samples_list, np.zeros_like(new_samples_list), marker='x', color='black')
        ax.set_xlabel('$x$')
        ax.set_ylabel('Density')
        ax.legend(loc='upper right')

        return line, scat, target_line, proposal_line
    
    ani = FuncAnimation(fig, update, frames=100, blit=True)
    ani.save('is.gif', writer='Pillow', fps=10)

    file_ = open("is.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )


# Gibbs Sampling Section
st.header("Gibbs Sampling")
st.markdown(r"""
**Mathematical Explanation**
         
Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method used to sample from a multivariate distribution. It constructs a Markov chain by sampling each variable in turn, conditional on the current values of the other variables.
            
This method is primarily used for sampling from multivariate distributions, particularly when direct sampling from the joint distribution is difficult due to complexity or high dimensionality.
The method is especially powerful for cases where it's easier to sample from the conditional distribution of each variable one at a time rather than from the joint distribution of all variables simultaneously.
 - Sample $ x_i $ from $ p(x_i | x/{x_i}) $ where $x/{x_i}$ represents the set of all random variables except $x_i$ itself. This is iteratively done for each variable $ x_i $

We need to specify a Joint Probability Distribution to conduct Gibbs Sampling.

**Advantages**
- Does not require tuning of parameters like step size, unlike other MCMC methods.
- Can be more efficient in high-dimensional spaces.

**Disadvantages**
- Can converge slowly if the variables are highly correlated.
- Requires the ability to sample from the conditional distribution of each variable.
""")

# Add dropdown for selecting the distribution type
dist_type = st.selectbox("Select Target Distribution", ["2D Gaussian", "2D Uniform"])

# Conditional inputs based on the distribution type selected
if dist_type == "2D Gaussian":
    mean_input_x1, mean_input_x2 = st.columns(2)
    
    mean_input_x1_val = mean_input_x1.number_input("Mean of $x_1$", value=0.0, format="%.2f")
    mean_input_x2_val = mean_input_x2.number_input("Mean of $x_2$", value=0.0, format="%.2f")

    # Input fields for covariance matrix
    cov_row1_col1, cov_row1_col2 = st.columns(2)
    cov_row2_col1, cov_row2_col2 = st.columns(2)

    var_x1 = cov_row1_col1.number_input("Variance of $x_1$", value=4.0, format="%.2f")
    cov_xy = cov_row1_col2.number_input("Covariance (σ₁₂ = σ₂₁)", value=1.0, format="%.2f")
    # Constrain the 2nd row, 1st column entry to be the same as the 1st row, 2nd column entry
    cov_row2_col1.number_input("Covariance (σ₂₁ = σ₁₂)", value=cov_xy, format="%.2f", key="cov_row2_col1")
    var_x2 = cov_row2_col2.number_input("Variance of $x_2$", value=1.0, format="%.2f")

    mean = np.array([mean_input_x1_val, mean_input_x2_val])
    cov = np.array([[var_x1, cov_xy], [cov_xy, var_x2]])
    
    target_dist = norm(loc=mean, scale=cov)
elif dist_type == "2D Uniform":

    a_input_x1, b_input_x1 = st.columns(2)
    a_input_x2, b_input_x2 = st.columns(2)

    a_input_x1_val = a_input_x1.number_input("Endpoint a of $x_1$", value=0.0, format="%.2f")
    b_input_x1_val = b_input_x1.number_input("Endpoint b of $x_1$", value=3.0, format="%.2f")

    a_input_x2_val = a_input_x2.number_input("Endpoint a of $x_2$", value=2.0, format="%.2f")
    b_input_x2_val = b_input_x2.number_input("Endpoint b of $x_2$", value=4.0, format="%.2f")

    a = np.array([a_input_x1_val, a_input_x2_val])
    b = np.array([b_input_x1_val, b_input_x2_val])
    # b = b-a

    # target_dist = uniform(loc=a, scale=b-a)

initial_state_x1, initial_state_x2 = st.columns(2)

initial_state_x1_val = initial_state_x1.number_input("Initial state for $x_1$", value=0.0, format="%.2f")
initial_state_x2_val = initial_state_x2.number_input("Initial state for $x_2$", value=0.0, format="%.2f")

initial_state_arr = np.array([initial_state_x1_val, initial_state_x2_val])
n_samples_gs = st.slider("Number of samples for Gibbs Sampling", 100, 5000, 1000)

if st.button('Run Gibbs Sampling'):
    is_normal = True if dist_type == "2D Gaussian" else False
    if(is_normal):
        samples_gs = gibbs_sampling_normal(initial_state_arr, n_samples_gs, mean, cov)  # Using a norm as a placeholder for the target distribution
    else:
        # samples_gs = gibbs_sampling_uniform(initial_state_arr, n_samples_gs, target_dist, is_normal)
        samples_gs = gibbs_sampling_uniform(initial_state_arr, n_samples_gs, a, b)
    # Plot for Gibbs Sampling
    st.write("Animation for Gibbs Sampling- might take a while to load!")
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='KDE of Accepted Samples')
    scat = ax.scatter([], [], marker='x', color='black', label='Accepted Samples')
    target_line, = ax.plot([], [], label='Target Distribution')
    proposal_line, = ax.plot([], [], label='Proposal Distribution')

    def update(frame):
        ax.clear()
        new_samples_list=samples_gs[:len(samples_gs)*(frame+1)//100]
        df_samples = pd.DataFrame(new_samples_list, columns=['x1', 'x2'])
        sns.kdeplot(data = df_samples, x="x1", y="x2", fill=True,color='blue', alpha=1)
        ax.scatter(new_samples_list[:, 0], new_samples_list[:, 1], alpha=1, marker='x', color='black', s=0.5, label='Samples')
        ax.set_xlim(-10, 10)
        ax.set_title(f"Gibbs Sampling: {len(new_samples_list)} Samples")
        ax.set_ylim(-10, 10)
        ax.legend(loc='upper right')
        # print(frame)
        return line, scat, target_line, proposal_line

    ani = FuncAnimation(fig, update, frames=100, blit=True)
    
    ani.save('gs.gif', writer='Pillow', fps=10)
    file_ = open("gs.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )