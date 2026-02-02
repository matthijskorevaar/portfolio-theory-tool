import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set page config
st.set_page_config(page_title="Portfolio Theory Tool", layout="wide")

# =============================================================================
# Presets (matching HTML version)
# =============================================================================
PRESETS = {
    "Default (Mixed)": {
        "stocks": [
            {"name": "Stock A", "mean": 10, "std": 15, "color": "#e74c3c"},
            {"name": "Stock B", "mean": 8, "std": 20, "color": "#3498db"},
            {"name": "Stock C", "mean": 12, "std": 25, "color": "#2ecc71"},
        ],
        "correlations": [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]
    },
    "CAPM Equilibrium": {
        "stocks": [
            {"name": "Defensive", "mean": 6.5, "std": 12, "color": "#e74c3c"},
            {"name": "Balanced", "mean": 10, "std": 18, "color": "#3498db"},
            {"name": "Aggressive", "mean": 13.5, "std": 24, "color": "#2ecc71"},
        ],
        "correlations": [[1.0, 0.6, 0.5], [0.6, 1.0, 0.7], [0.5, 0.7, 1.0]]
    },
    "Diverse Assets": {
        "stocks": [
            {"name": "Bonds", "mean": 5, "std": 8, "color": "#e74c3c"},
            {"name": "Stocks", "mean": 12, "std": 20, "color": "#3498db"},
            {"name": "Real Estate", "mean": 9, "std": 15, "color": "#2ecc71"},
        ],
        "correlations": [[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]]
    },
    "High Correlation": {
        "stocks": [
            {"name": "Tech A", "mean": 15, "std": 25, "color": "#e74c3c"},
            {"name": "Tech B", "mean": 14, "std": 24, "color": "#3498db"},
            {"name": "Tech C", "mean": 16, "std": 26, "color": "#2ecc71"},
        ],
        "correlations": [[1.0, 0.85, 0.80], [0.85, 1.0, 0.82], [0.80, 0.82, 1.0]]
    },
    "Negative Correlation": {
        "stocks": [
            {"name": "Stocks", "mean": 12, "std": 20, "color": "#e74c3c"},
            {"name": "Gold", "mean": 6, "std": 15, "color": "#3498db"},
            {"name": "Bonds", "mean": 4, "std": 8, "color": "#2ecc71"},
        ],
        "correlations": [[1.0, -0.3, -0.1], [-0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]]
    }
}

# =============================================================================
# Initialize Session State
# =============================================================================
if 'mu_assets' not in st.session_state:
    st.session_state.mu_assets = [10.0, 8.0, 12.0]
if 'sigma_assets' not in st.session_state:
    st.session_state.sigma_assets = [15.0, 20.0, 25.0]
if 'asset_names' not in st.session_state:
    st.session_state.asset_names = ["Stock A", "Stock B", "Stock C"]
if 'correlations' not in st.session_state:
    st.session_state.correlations = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]
if 'shock_history' not in st.session_state:
    st.session_state.shock_history = []
if 'original_mu' not in st.session_state:
    st.session_state.original_mu = None
if 'lock_axes' not in st.session_state:
    st.session_state.lock_axes = False
if 'axis_bounds' not in st.session_state:
    st.session_state.axis_bounds = None
if 'animation_step' not in st.session_state:
    st.session_state.animation_step = 0
if 'animating_stock' not in st.session_state:
    st.session_state.animating_stock = None

# =============================================================================
# Helper Functions
# =============================================================================
def apply_preset(preset_name):
    preset = PRESETS[preset_name]
    n = len(preset["stocks"])
    st.session_state.mu_assets = [s["mean"] for s in preset["stocks"]]
    st.session_state.sigma_assets = [s["std"] for s in preset["stocks"]]
    st.session_state.asset_names = [s["name"] for s in preset["stocks"]]
    st.session_state.correlations = [row[:] for row in preset["correlations"]]
    st.session_state.original_mu = st.session_state.mu_assets.copy()
    st.session_state.shock_history = []

def portfolio_return(weights, mu):
    return np.dot(weights, mu)

def portfolio_std(weights, cov):
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

def neg_sharpe_ratio(weights, mu, cov, rf):
    ret = portfolio_return(weights, mu)
    std = portfolio_std(weights, cov)
    if std < 1e-10:
        return 1e10
    return -(ret - rf) / std

def minimize_variance(target_return, mu, cov, n):
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_return}
    ]
    bounds = tuple((-0.5, 1.5) for _ in range(n))
    init_weights = np.ones(n) / n
    result = minimize(lambda w: portfolio_std(w, cov)**2, init_weights,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def find_tangency_portfolio(mu, cov, rf, n):
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((-0.5, 1.5) for _ in range(n))
    init_weights = np.ones(n) / n
    result = minimize(neg_sharpe_ratio, init_weights, args=(mu, cov, rf),
                     method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def build_cov_matrix(sigmas, corr_matrix):
    n = len(sigmas)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = sigmas[i] * sigmas[j] * corr_matrix[i][j]
    return cov

# =============================================================================
# Main App
# =============================================================================
st.title("📊 Portfolio Theory & CAPM Interactive Tool")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Preset selector
preset_choice = st.sidebar.selectbox("Load Preset", list(PRESETS.keys()))
if st.sidebar.button("Apply Preset"):
    apply_preset(preset_choice)
    st.rerun()

st.sidebar.markdown("---")

# Risk-free rate
rf = st.sidebar.number_input("Risk-Free Rate (%)", value=3.0, step=0.25) / 100

# Number of assets
n_assets = st.sidebar.selectbox("Number of Assets", [2, 3, 4], index=1)

# Ensure arrays are correct length
while len(st.session_state.mu_assets) < n_assets:
    st.session_state.mu_assets.append(8.0)
    st.session_state.sigma_assets.append(15.0)
    st.session_state.asset_names.append(f"Stock {len(st.session_state.mu_assets)}")
while len(st.session_state.correlations) < n_assets:
    st.session_state.correlations.append([0.3] * n_assets)
    for row in st.session_state.correlations:
        while len(row) < n_assets:
            row.append(0.3)

# Asset parameters in sidebar
st.sidebar.subheader("Asset Parameters")
for i in range(n_assets):
    with st.sidebar.expander(f"{st.session_state.asset_names[i]}", expanded=(i == 0)):
        st.session_state.asset_names[i] = st.text_input(f"Name", value=st.session_state.asset_names[i], key=f"name_{i}")
        st.session_state.mu_assets[i] = st.number_input(f"Expected Return (%)", value=float(st.session_state.mu_assets[i]), step=0.5, key=f"mu_{i}")
        st.session_state.sigma_assets[i] = st.number_input(f"Std Dev (%)", value=float(st.session_state.sigma_assets[i]), step=1.0, min_value=1.0, key=f"sigma_{i}")

# Correlations
st.sidebar.subheader("Correlations")
for i in range(n_assets):
    for j in range(i+1, n_assets):
        val = st.sidebar.slider(
            f"ρ({st.session_state.asset_names[i][:6]}, {st.session_state.asset_names[j][:6]})",
            min_value=-0.9, max_value=0.9,
            value=float(st.session_state.correlations[i][j]),
            step=0.1, key=f"corr_{i}_{j}"
        )
        st.session_state.correlations[i][j] = val
        st.session_state.correlations[j][i] = val

# Set diagonal to 1
for i in range(n_assets):
    st.session_state.correlations[i][i] = 1.0

# =============================================================================
# News Shocks Panel
# =============================================================================
st.subheader("📰 News Shocks")

col_news = st.columns(n_assets + 2)

for i in range(n_assets):
    with col_news[i]:
        st.markdown(f"**{st.session_state.asset_names[i]}**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"📈+2%", key=f"up_{i}"):
                st.session_state.mu_assets[i] += 2.0
                st.session_state.shock_history.append(f"📈 {st.session_state.asset_names[i]} +2%")
                st.rerun()
        with c2:
            if st.button(f"📉-2%", key=f"down_{i}"):
                st.session_state.mu_assets[i] -= 2.0
                st.session_state.shock_history.append(f"📉 {st.session_state.asset_names[i]} -2%")
                st.rerun()

with col_news[n_assets]:
    st.markdown("**Options**")
    st.session_state.lock_axes = st.checkbox("🔒 Lock Axes", value=st.session_state.lock_axes)

with col_news[n_assets + 1]:
    st.markdown("**Reset**")
    if st.button("🔄 Reset"):
        if st.session_state.original_mu:
            st.session_state.mu_assets = st.session_state.original_mu.copy()
        st.session_state.shock_history = []
        st.session_state.animation_step = 0
        st.session_state.animating_stock = None
        st.rerun()

# Store original if not set
if st.session_state.original_mu is None:
    st.session_state.original_mu = st.session_state.mu_assets.copy()

# Show shock history
if st.session_state.shock_history:
    with st.expander("📜 News History"):
        for shock in st.session_state.shock_history[-10:]:
            st.write(shock)

# =============================================================================
# Calculations
# =============================================================================
mu_assets = np.array(st.session_state.mu_assets[:n_assets]) / 100
sigmas = np.array(st.session_state.sigma_assets[:n_assets]) / 100
corr_matrix = [row[:n_assets] for row in st.session_state.correlations[:n_assets]]
cov_matrix = build_cov_matrix(sigmas, corr_matrix)

# Efficient frontier
min_ret = mu_assets.min() - 0.02
max_ret = mu_assets.max() + 0.05
target_returns = np.linspace(min_ret, max_ret, 50)

frontier_stds = []
frontier_rets = []

for target in target_returns:
    result = minimize_variance(target, mu_assets, cov_matrix, n_assets)
    if result.success:
        frontier_stds.append(portfolio_std(result.x, cov_matrix))
        frontier_rets.append(target)

frontier_stds = np.array(frontier_stds)
frontier_rets = np.array(frontier_rets)

# Minimum variance portfolio
min_var_idx = np.argmin(frontier_stds)
min_var_std = frontier_stds[min_var_idx]
min_var_ret = frontier_rets[min_var_idx]

# Tangency portfolio
tangency_weights = find_tangency_portfolio(mu_assets, cov_matrix, rf, n_assets)
tangency_ret = portfolio_return(tangency_weights, mu_assets)
tangency_std = portfolio_std(tangency_weights, cov_matrix)
sharpe_ratio = (tangency_ret - rf) / tangency_std if tangency_std > 0 else 0

# Betas
betas = []
for i in range(n_assets):
    cov_with_market = np.dot(cov_matrix[i, :], tangency_weights)
    var_market = np.dot(tangency_weights.T, np.dot(cov_matrix, tangency_weights))
    beta = cov_with_market / var_market if var_market > 0 else 1.0
    betas.append(beta)
betas = np.array(betas)

# Alphas
alphas = mu_assets - (rf + betas * (tangency_ret - rf))

# =============================================================================
# Axis bounds (for lock axes feature)
# =============================================================================
if st.session_state.lock_axes and st.session_state.axis_bounds is None:
    st.session_state.axis_bounds = {
        'markowitz': {'xmin': 0, 'xmax': max(sigmas) * 1.8, 'ymin': min(rf - 0.02, min_ret), 'ymax': max_ret * 1.2},
        'capm': {'xmin': -0.1, 'xmax': max(betas) + 0.5, 'ymin': min(rf - 0.02, min(mu_assets) - 0.02), 'ymax': max(mu_assets) * 1.2}
    }

# =============================================================================
# Plots
# =============================================================================
col_left, col_right = st.columns(2)

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

# Left: Efficient Frontier
with col_left:
    st.subheader("Efficient Frontier & CML")

    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Efficient frontier
    efficient_mask = frontier_rets >= min_var_ret
    ax1.plot(frontier_stds[efficient_mask] * 100, frontier_rets[efficient_mask] * 100,
            'b-', linewidth=2.5, label='Efficient Frontier')

    # Inefficient part
    inefficient_mask = frontier_rets <= min_var_ret
    ax1.plot(frontier_stds[inefficient_mask] * 100, frontier_rets[inefficient_mask] * 100,
            'b--', linewidth=1.5, alpha=0.5)

    # Capital Market Line
    cml_stds = np.linspace(0, max(frontier_stds) * 1.3, 50)
    cml_rets = rf + sharpe_ratio * cml_stds
    ax1.plot(cml_stds * 100, cml_rets * 100, 'g-', linewidth=2, label='CML')

    # Individual assets
    for i in range(n_assets):
        ax1.scatter(sigmas[i] * 100, mu_assets[i] * 100, s=150, c=colors[i],
                   marker='o', label=st.session_state.asset_names[i], zorder=5, edgecolor='black', linewidth=1.5)

    # Tangency portfolio
    ax1.scatter(tangency_std * 100, tangency_ret * 100, s=250, c='gold', marker='*',
               label='Market Portfolio', zorder=10, edgecolor='black', linewidth=1.5)

    # Risk-free
    ax1.scatter(0, rf * 100, s=100, c='green', marker='s', label=f'Risk-Free ({rf*100:.1f}%)', zorder=5)

    ax1.set_xlabel('Standard Deviation (%)', fontsize=12)
    ax1.set_ylabel('Expected Return (%)', fontsize=12)
    ax1.set_title('Mean-Variance Frontier', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Apply locked axes if enabled
    if st.session_state.lock_axes and st.session_state.axis_bounds:
        ax1.set_xlim(st.session_state.axis_bounds['markowitz']['xmin'] * 100,
                     st.session_state.axis_bounds['markowitz']['xmax'] * 100)
        ax1.set_ylim(st.session_state.axis_bounds['markowitz']['ymin'] * 100,
                     st.session_state.axis_bounds['markowitz']['ymax'] * 100)
    else:
        ax1.set_xlim(0, max(frontier_stds) * 120 + 5)
        ax1.set_ylim(min(rf - 0.01, min_ret) * 100, max_ret * 100 + 2)

    st.pyplot(fig1)

# Right: SML
with col_right:
    st.subheader("Security Market Line (SML)")

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # SML
    beta_range = np.linspace(-0.2, max(betas) + 0.5, 50)
    sml_returns = rf + beta_range * (tangency_ret - rf)
    ax2.plot(beta_range, sml_returns * 100, 'r-', linewidth=2.5, label='SML')

    # Individual assets
    for i in range(n_assets):
        # Color based on alpha
        if alphas[i] > 0.005:
            color = '#27ae60'  # Green (undervalued)
            marker_label = f"{st.session_state.asset_names[i]} (α>0)"
        elif alphas[i] < -0.005:
            color = '#e74c3c'  # Red (overvalued)
            marker_label = f"{st.session_state.asset_names[i]} (α<0)"
        else:
            color = colors[i]
            marker_label = st.session_state.asset_names[i]

        ax2.scatter(betas[i], mu_assets[i] * 100, s=150, c=color,
                   marker='o', label=marker_label, zorder=5, edgecolor='black', linewidth=1.5)

        # Draw alpha line
        expected_capm = (rf + betas[i] * (tangency_ret - rf)) * 100
        if abs(alphas[i]) > 0.005:
            ax2.plot([betas[i], betas[i]], [expected_capm, mu_assets[i] * 100],
                    '--', color=color, linewidth=2, alpha=0.7)

    # Market portfolio
    ax2.scatter(1.0, tangency_ret * 100, s=250, c='gold', marker='*',
               label='Market Portfolio', zorder=10, edgecolor='black', linewidth=1.5)

    # Risk-free
    ax2.scatter(0, rf * 100, s=100, c='green', marker='s', label='Risk-Free', zorder=5)

    ax2.set_xlabel('Beta (β)', fontsize=12)
    ax2.set_ylabel('Expected Return (%)', fontsize=12)
    ax2.set_title('Security Market Line', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=rf * 100, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)

    # Apply locked axes if enabled
    if st.session_state.lock_axes and st.session_state.axis_bounds:
        ax2.set_xlim(st.session_state.axis_bounds['capm']['xmin'],
                     st.session_state.axis_bounds['capm']['xmax'])
        ax2.set_ylim(st.session_state.axis_bounds['capm']['ymin'] * 100,
                     st.session_state.axis_bounds['capm']['ymax'] * 100)
    else:
        ax2.set_xlim(-0.2, max(betas) + 0.3)
        ax2.set_ylim(min(rf - 0.01, min(mu_assets) - 0.02) * 100, max(mu_assets) * 100 + 3)

    st.pyplot(fig2)

# =============================================================================
# Summary Statistics
# =============================================================================
st.subheader("📊 Portfolio Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Market Portfolio Return", f"{tangency_ret:.2%}")
col2.metric("Market Portfolio Risk", f"{tangency_std:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Market Risk Premium", f"{(tangency_ret - rf):.2%}")

# Asset details table
st.subheader("Asset Details")
asset_df = pd.DataFrame({
    'Asset': st.session_state.asset_names[:n_assets],
    'E[R] (%)': [f"{m*100:.1f}" for m in mu_assets],
    'σ (%)': [f"{s*100:.1f}" for s in sigmas],
    'Beta (β)': [f"{b:.2f}" for b in betas],
    'CAPM E[R] (%)': [f"{(rf + b*(tangency_ret-rf))*100:.1f}" for b in betas],
    'Alpha (%)': [f"{a*100:.2f}" for a in alphas],
    'Weight in M (%)': [f"{w*100:.1f}" for w in tangency_weights]
})
st.dataframe(asset_df, use_container_width=True, hide_index=True)

# =============================================================================
# Equilibrium Animation
# =============================================================================
st.subheader("⚖️ Equilibrium Animation")
st.markdown("Watch how the market returns to equilibrium after a news shock:")

anim_col1, anim_col2 = st.columns([3, 1])

with anim_col1:
    anim_stock = st.selectbox("Select stock for animation:", st.session_state.asset_names[:n_assets])
    anim_idx = st.session_state.asset_names[:n_assets].index(anim_stock)

with anim_col2:
    if st.button("▶️ Start Animation"):
        st.session_state.animating_stock = anim_idx
        st.session_state.animation_step = 1
        # Apply initial shock
        st.session_state.mu_assets[anim_idx] += 4.0
        st.session_state.shock_history.append(f"⚖️ Animation: {anim_stock} +4%")
        st.rerun()

# Animation steps
if st.session_state.animation_step > 0 and st.session_state.animating_stock is not None:
    idx = st.session_state.animating_stock
    stock_name = st.session_state.asset_names[idx]

    steps = {
        1: f"📢 **Step 1**: News shock! {stock_name} expected return jumps (now above SML = positive alpha)",
        2: f"💡 **Step 2**: {stock_name} is UNDERVALUED. Investors want to buy it!",
        3: f"📈 **Step 3**: Buying pressure increases price → Expected return falls",
        4: f"⚖️ **Step 4**: Equilibrium restored! {stock_name} back on SML"
    }

    st.info(steps.get(st.session_state.animation_step, "Animation complete"))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("➡️ Next Step"):
            st.session_state.animation_step += 1
            if st.session_state.animation_step == 3:
                # Partial adjustment
                target = rf + betas[idx] * (tangency_ret - rf)
                current = st.session_state.mu_assets[idx] / 100
                st.session_state.mu_assets[idx] = ((current + target) / 2) * 100
            elif st.session_state.animation_step == 4:
                # Full equilibrium
                target = rf + betas[idx] * (tangency_ret - rf)
                st.session_state.mu_assets[idx] = target * 100
            elif st.session_state.animation_step > 4:
                st.session_state.animation_step = 0
                st.session_state.animating_stock = None
            st.rerun()

    with col_b:
        if st.button("⏹️ Stop"):
            st.session_state.animation_step = 0
            st.session_state.animating_stock = None
            st.rerun()

    with col_c:
        if st.button("🔄 Reset & Stop"):
            st.session_state.animation_step = 0
            st.session_state.animating_stock = None
            if st.session_state.original_mu:
                st.session_state.mu_assets = st.session_state.original_mu.copy()
            st.rerun()

# =============================================================================
# Educational Notes
# =============================================================================
with st.expander("📚 Key Concepts"):
    st.markdown("""
    **Efficient Frontier**: The set of portfolios offering the highest expected return for each level of risk.

    **Capital Market Line (CML)**: When a risk-free asset exists, optimal portfolios lie on the line
    from the risk-free rate tangent to the efficient frontier. The tangent point is the **Market Portfolio**.

    **Security Market Line (SML)**: Shows the relationship between systematic risk (beta) and expected return.
    - Assets **above** the SML are undervalued (positive alpha) → buy pressure → price rises → return falls
    - Assets **below** the SML are overvalued (negative alpha) → sell pressure → price falls → return rises

    **News Shock Effects**:
    - Positive news increases expected return → stock moves above SML
    - Arbitrage forces push it back to equilibrium
    - The efficient frontier shifts when expected returns change
    """)

with st.expander("📐 Formulas"):
    st.latex(r"\text{Sharpe Ratio} = \frac{E[R_M] - R_f}{\sigma_M}")
    st.latex(r"\beta_i = \frac{\text{Cov}(R_i, R_M)}{\text{Var}(R_M)}")
    st.latex(r"\text{CAPM: } E[R_i] = R_f + \beta_i (E[R_M] - R_f)")
    st.latex(r"\alpha_i = E[R_i] - [R_f + \beta_i (E[R_M] - R_f)]")
