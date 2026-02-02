import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set page config
st.set_page_config(page_title="Portfolio Visualizer", layout="wide")

# Create tabs for different views
tab1, tab2 = st.tabs(["Two-Asset Portfolio", "Markowitz & News Shocks"])

# =============================================================================
# TAB 1: Original Two-Asset Portfolio Visualizer
# =============================================================================
with tab1:
    st.title("Two-Asset Portfolio Visualizer")
    st.markdown("""
    This interactive tool allows you to explore the relationship between portfolio risk (standard deviation)
    and return for a two-asset portfolio. Adjust the parameters in the sidebar to see how
    diversification affects the efficient frontier.
    """)

    # Sidebar inputs for Tab 1
    st.sidebar.header("Two-Asset Parameters")

    st.sidebar.subheader("Asset 1")
    name1 = st.sidebar.text_input("Name", value="Stock A", key="n1")
    mu1 = st.sidebar.number_input(f"Expected Return (%)", value=10.0, step=0.5, key="m1") / 100
    sigma1 = st.sidebar.number_input(f"Standard Deviation (%)", value=20.0, step=1.0, min_value=0.0, key="s1") / 100

    st.sidebar.subheader("Asset 2")
    name2 = st.sidebar.text_input("Name", value="Stock B", key="n2")
    mu2 = st.sidebar.number_input(f"Expected Return (%)", value=6.0, step=0.5, key="m2") / 100
    sigma2 = st.sidebar.number_input(f"Standard Deviation (%)", value=12.0, step=1.0, min_value=0.0, key="s2") / 100

    st.sidebar.subheader("Correlation")
    rho = st.sidebar.slider("Correlation Coefficient (ρ)", min_value=-1.0, max_value=1.0, value=0.2, step=0.05)

    # Calculations
    weights_1 = np.linspace(0, 1, 100)
    weights_2 = 1 - weights_1

    # Portfolio Return
    port_returns = weights_1 * mu1 + weights_2 * mu2

    # Portfolio Variance
    port_vars = (weights_1**2 * sigma1**2) + (weights_2**2 * sigma2**2) + (2 * weights_1 * weights_2 * sigma1 * sigma2 * rho)
    port_stds = np.sqrt(port_vars)

    # Interactive Weight Selection
    st.subheader("Explore Allocations")
    selected_w1 = st.slider(f"Weight in {name1} (%)", min_value=0, max_value=100, value=50, step=1, key="tab1_w1") / 100
    selected_w2 = 1 - selected_w1

    # Calculate specific point
    sel_ret = selected_w1 * mu1 + selected_w2 * mu2
    sel_var = (selected_w1**2 * sigma1**2) + (selected_w2**2 * sigma2**2) + (2 * selected_w1 * selected_w2 * sigma1 * sigma2 * rho)
    sel_std = np.sqrt(sel_var)

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Portfolio Return", f"{sel_ret:.2%}")
    col2.metric(f"Portfolio Risk (Std Dev)", f"{sel_std:.2%}")
    col3.metric("Diversification Benefit", "Yes" if sel_std < (selected_w1*sigma1 + selected_w2*sigma2) else "No (Linear)")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(port_stds * 100, port_returns * 100, label="Efficient Frontier", color="#003057", linewidth=2)
    ax.scatter(sigma1 * 100, mu1 * 100, color="green", s=100, label=name1, zorder=5)
    ax.scatter(sigma2 * 100, mu2 * 100, color="orange", s=100, label=name2, zorder=5)
    ax.scatter(sel_std * 100, sel_ret * 100, color="red", s=150, marker='*', label="Selected Portfolio", zorder=10)
    ax.set_title(f"Portfolio Risk vs. Return (Correlation = {rho})", fontsize=14)
    ax.set_xlabel("Standard Deviation (%)", fontsize=12)
    ax.set_ylabel("Expected Return (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

    with st.expander("Show Formula Details"):
        st.latex(r"E(R_p) = w_1 E(R_1) + w_2 E(R_2)")
        st.latex(r"\sigma_p = \sqrt{w_1^2 \sigma_1^2 + w_2^2 \sigma_2^2 + 2 w_1 w_2 \sigma_1 \sigma_2 \rho_{1,2}}")


# =============================================================================
# TAB 2: Markowitz & News Shocks
# =============================================================================
with tab2:
    st.title("Markowitz Portfolio Theory & News Shocks")
    st.markdown("""
    Explore how **news about expected returns** shifts the efficient frontier, market portfolio, and Security Market Line (SML).

    Use the **News Shock** buttons to simulate positive or negative news about each stock and observe how:
    - The **Efficient Frontier** shifts
    - The **Market Portfolio** (tangency portfolio) changes
    - The **Security Market Line** rotates
    """)

    # Initialize session state for expected returns (to handle news shocks)
    if 'mu_assets' not in st.session_state:
        st.session_state.mu_assets = [8.0, 12.0, 6.0]  # Default expected returns (%)
    if 'shock_history' not in st.session_state:
        st.session_state.shock_history = []

    # Sidebar for Markowitz parameters
    st.sidebar.markdown("---")
    st.sidebar.header("Markowitz Parameters")

    # Number of assets
    n_assets = st.sidebar.selectbox("Number of Assets", [2, 3, 4], index=1, key="n_assets")

    # Risk-free rate
    rf = st.sidebar.number_input("Risk-Free Rate (%)", value=2.0, step=0.25, key="rf") / 100

    # Asset names
    default_names = ["Stock A", "Stock B", "Stock C", "Stock D"]
    asset_names = []
    for i in range(n_assets):
        name = st.sidebar.text_input(f"Asset {i+1} Name", value=default_names[i], key=f"mk_name_{i}")
        asset_names.append(name)

    # Extend mu_assets if needed
    while len(st.session_state.mu_assets) < n_assets:
        st.session_state.mu_assets.append(5.0)

    # Asset parameters
    st.sidebar.subheader("Expected Returns (%)")
    st.sidebar.caption("Use News Shock buttons in main panel to change these")

    # Display current expected returns (read-only display)
    for i in range(n_assets):
        st.sidebar.write(f"{asset_names[i]}: {st.session_state.mu_assets[i]:.1f}%")

    # Standard deviations
    st.sidebar.subheader("Standard Deviations (%)")
    sigmas = []
    default_sigmas = [15.0, 25.0, 20.0, 18.0]
    for i in range(n_assets):
        sigma = st.sidebar.number_input(f"{asset_names[i]}", value=default_sigmas[i], step=1.0, min_value=1.0, key=f"mk_sigma_{i}") / 100
        sigmas.append(sigma)

    # Correlation matrix
    st.sidebar.subheader("Correlations")
    corr_matrix = np.eye(n_assets)
    corr_inputs = {}
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            key = f"corr_{i}_{j}"
            default_corr = 0.3 if i != j else 1.0
            corr_val = st.sidebar.slider(f"ρ({asset_names[i]}, {asset_names[j]})",
                                         min_value=-0.9, max_value=0.9, value=default_corr, step=0.1, key=key)
            corr_matrix[i, j] = corr_val
            corr_matrix[j, i] = corr_val

    # Build covariance matrix
    sigmas_arr = np.array(sigmas)
    cov_matrix = np.outer(sigmas_arr, sigmas_arr) * corr_matrix

    # Get current expected returns
    mu_assets = np.array(st.session_state.mu_assets[:n_assets]) / 100

    # =============================================================================
    # News Shock Controls
    # =============================================================================
    st.subheader("📰 News Shocks")
    st.markdown("Simulate news events that change expected returns:")

    # Create columns for news shock buttons
    cols = st.columns(n_assets + 1)

    for i in range(n_assets):
        with cols[i]:
            st.markdown(f"**{asset_names[i]}**")
            col_up, col_down = st.columns(2)
            with col_up:
                if st.button(f"📈 +2%", key=f"up_{i}"):
                    st.session_state.mu_assets[i] += 2.0
                    st.session_state.shock_history.append(f"📈 {asset_names[i]} +2%")
                    st.rerun()
            with col_down:
                if st.button(f"📉 -2%", key=f"down_{i}"):
                    st.session_state.mu_assets[i] -= 2.0
                    st.session_state.shock_history.append(f"📉 {asset_names[i]} -2%")
                    st.rerun()

    with cols[n_assets]:
        st.markdown("**Reset**")
        if st.button("🔄 Reset All", key="reset"):
            st.session_state.mu_assets = [8.0, 12.0, 6.0, 10.0]
            st.session_state.shock_history = []
            st.rerun()

    # Show shock history
    if st.session_state.shock_history:
        with st.expander("News History"):
            for shock in st.session_state.shock_history[-10:]:
                st.write(shock)

    # =============================================================================
    # Portfolio Optimization Functions
    # =============================================================================
    def portfolio_return(weights, mu):
        return np.dot(weights, mu)

    def portfolio_std(weights, cov):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    def neg_sharpe_ratio(weights, mu, cov, rf):
        ret = portfolio_return(weights, mu)
        std = portfolio_std(weights, cov)
        return -(ret - rf) / std

    def minimize_variance(target_return, mu, cov, n):
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu) - target_return}
        ]
        bounds = tuple((-0.5, 1.5) for _ in range(n))  # Allow some short selling
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

    # =============================================================================
    # Compute Efficient Frontier
    # =============================================================================
    # Target returns range
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

    # Find minimum variance portfolio
    min_var_idx = np.argmin(frontier_stds)
    min_var_std = frontier_stds[min_var_idx]
    min_var_ret = frontier_rets[min_var_idx]

    # Find tangency (market) portfolio
    tangency_weights = find_tangency_portfolio(mu_assets, cov_matrix, rf, n_assets)
    tangency_ret = portfolio_return(tangency_weights, mu_assets)
    tangency_std = portfolio_std(tangency_weights, cov_matrix)
    sharpe_ratio = (tangency_ret - rf) / tangency_std

    # Compute betas for each asset
    betas = []
    for i in range(n_assets):
        # Beta = Cov(Ri, Rm) / Var(Rm)
        cov_with_market = np.dot(cov_matrix[i, :], tangency_weights)
        var_market = np.dot(tangency_weights.T, np.dot(cov_matrix, tangency_weights))
        beta = cov_with_market / var_market
        betas.append(beta)
    betas = np.array(betas)

    # =============================================================================
    # Plotting
    # =============================================================================
    col_left, col_right = st.columns(2)

    # Left plot: Efficient Frontier with CML
    with col_left:
        st.subheader("Efficient Frontier & Capital Market Line")

        fig1, ax1 = plt.subplots(figsize=(8, 6))

        # Plot efficient frontier (only upper part)
        efficient_mask = frontier_rets >= min_var_ret
        ax1.plot(frontier_stds[efficient_mask] * 100, frontier_rets[efficient_mask] * 100,
                'b-', linewidth=2.5, label='Efficient Frontier')

        # Plot inefficient part (dashed)
        inefficient_mask = frontier_rets <= min_var_ret
        ax1.plot(frontier_stds[inefficient_mask] * 100, frontier_rets[inefficient_mask] * 100,
                'b--', linewidth=1.5, alpha=0.5, label='Inefficient Frontier')

        # Plot Capital Market Line
        cml_stds = np.linspace(0, max(frontier_stds) * 1.2, 50)
        cml_rets = rf + sharpe_ratio * cml_stds
        ax1.plot(cml_stds * 100, cml_rets * 100, 'g-', linewidth=2, label='Capital Market Line')

        # Plot individual assets
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        for i in range(n_assets):
            ax1.scatter(sigmas[i] * 100, mu_assets[i] * 100, s=150, c=colors[i],
                       marker='o', label=asset_names[i], zorder=5, edgecolor='black', linewidth=1.5)

        # Plot tangency portfolio
        ax1.scatter(tangency_std * 100, tangency_ret * 100, s=250, c='gold', marker='*',
                   label='Market Portfolio', zorder=10, edgecolor='black', linewidth=1.5)

        # Plot risk-free rate
        ax1.scatter(0, rf * 100, s=100, c='green', marker='s', label=f'Risk-Free ({rf*100:.1f}%)', zorder=5)

        # Plot minimum variance portfolio
        ax1.scatter(min_var_std * 100, min_var_ret * 100, s=100, c='purple', marker='D',
                   label='Min Variance', zorder=5, edgecolor='black')

        ax1.set_xlabel('Standard Deviation (%)', fontsize=12)
        ax1.set_ylabel('Expected Return (%)', fontsize=12)
        ax1.set_title('Mean-Variance Frontier', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(frontier_stds) * 120 + 5)
        ax1.set_ylim(min(rf - 0.01, min_ret) * 100, max_ret * 100 + 2)

        st.pyplot(fig1)

    # Right plot: Security Market Line
    with col_right:
        st.subheader("Security Market Line (SML)")

        fig2, ax2 = plt.subplots(figsize=(8, 6))

        # Plot SML
        beta_range = np.linspace(-0.2, 2.0, 50)
        sml_returns = rf + beta_range * (tangency_ret - rf)
        ax2.plot(beta_range, sml_returns * 100, 'b-', linewidth=2.5, label='Security Market Line')

        # Plot individual assets
        for i in range(n_assets):
            ax2.scatter(betas[i], mu_assets[i] * 100, s=150, c=colors[i],
                       marker='o', label=asset_names[i], zorder=5, edgecolor='black', linewidth=1.5)

            # Show if overvalued/undervalued
            expected_by_capm = rf + betas[i] * (tangency_ret - rf)
            if mu_assets[i] > expected_by_capm + 0.005:
                ax2.annotate('▲', (betas[i], mu_assets[i] * 100 + 0.5), fontsize=12, ha='center', color='green')
            elif mu_assets[i] < expected_by_capm - 0.005:
                ax2.annotate('▼', (betas[i], mu_assets[i] * 100 + 0.5), fontsize=12, ha='center', color='red')

        # Plot market portfolio (beta = 1)
        ax2.scatter(1.0, tangency_ret * 100, s=250, c='gold', marker='*',
                   label='Market Portfolio', zorder=10, edgecolor='black', linewidth=1.5)

        # Plot risk-free (beta = 0)
        ax2.scatter(0, rf * 100, s=100, c='green', marker='s', label=f'Risk-Free', zorder=5)

        ax2.set_xlabel('Beta (β)', fontsize=12)
        ax2.set_ylabel('Expected Return (%)', fontsize=12)
        ax2.set_title('Security Market Line', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=rf * 100, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
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
    col4.metric("Risk Premium", f"{(tangency_ret - rf):.2%}")

    # Asset details table
    st.subheader("Asset Details")
    asset_df = pd.DataFrame({
        'Asset': asset_names,
        'E[R] (%)': [f"{m*100:.1f}" for m in mu_assets],
        'σ (%)': [f"{s*100:.1f}" for s in sigmas],
        'Beta (β)': [f"{b:.2f}" for b in betas],
        'CAPM E[R] (%)': [f"{(rf + b*(tangency_ret-rf))*100:.1f}" for b in betas],
        'Alpha (%)': [f"{(mu_assets[i] - (rf + betas[i]*(tangency_ret-rf)))*100:.1f}" for i in range(n_assets)],
        'Weight in M (%)': [f"{w*100:.1f}" for w in tangency_weights]
    })
    st.dataframe(asset_df, use_container_width=True, hide_index=True)

    # Educational notes
    with st.expander("📚 Key Concepts"):
        st.markdown("""
        **Efficient Frontier**: The set of portfolios offering the highest expected return for each level of risk.

        **Capital Market Line (CML)**: When a risk-free asset exists, the optimal portfolios lie on the line
        from the risk-free rate tangent to the efficient frontier. This tangent portfolio is the **Market Portfolio**.

        **Security Market Line (SML)**: Shows the relationship between systematic risk (beta) and expected return.
        - Assets **above** the SML are undervalued (positive alpha) ▲
        - Assets **below** the SML are overvalued (negative alpha) ▼

        **News Shock Effects**:
        - Positive news about a stock increases its expected return
        - This shifts the efficient frontier outward
        - The market portfolio composition changes (more weight in the "good news" stock)
        - The SML rotates if the market risk premium changes
        """)

    with st.expander("📐 Formulas"):
        st.latex(r"\text{Sharpe Ratio} = \frac{E[R_M] - R_f}{\sigma_M}")
        st.latex(r"\beta_i = \frac{\text{Cov}(R_i, R_M)}{\text{Var}(R_M)}")
        st.latex(r"\text{CAPM: } E[R_i] = R_f + \beta_i (E[R_M] - R_f)")
        st.latex(r"\alpha_i = E[R_i] - [R_f + \beta_i (E[R_M] - R_f)]")
