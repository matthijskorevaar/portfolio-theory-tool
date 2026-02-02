# Portfolio Theory & CAPM Interactive Tool

An interactive educational tool for teaching Markowitz portfolio theory, the efficient frontier, and the Capital Asset Pricing Model (CAPM).

## Live Demo

**HTML Version (GitHub Pages):** [https://matthijskorevaar.github.io/portfolio-theory-tool/](https://matthijskorevaar.github.io/portfolio-theory-tool/)

**Streamlit Version:** Deploy your own using the instructions below.

## Features

### Markowitz View
- Visualize the efficient frontier with 2-4 assets
- See how correlation affects diversification benefits
- Capital Market Line (CML) with risk-free asset
- Tangency (market) portfolio identification
- Utility curves for different risk aversion levels

### CAPM View
- Security Market Line (SML)
- Beta calculation for each asset
- Alpha (mispricing) identification
- Visual indicators for over/undervalued assets

### News Shocks (NEW!)
- 📈 Simulate positive news (+2% expected return)
- 📉 Simulate negative news (-2% expected return)
- Watch how the efficient frontier shifts
- See the market portfolio composition change
- Observe SML rotation when market risk premium changes
- Reset button to restore original values

## Two Versions Available

### 1. HTML Version (`index.html`)
- No installation required
- Works directly in browser
- Hosted via GitHub Pages

### 2. Streamlit Version (`portfolio_app.py`)
- More interactive controls
- Additional features
- Requires Python environment

## Deploying on Streamlit Cloud

1. Fork this repository or use it directly
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click "New app"
4. Select this repository
5. Set main file path: `portfolio_app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-username]-portfolio-theory-tool-portfolio-app-[id].streamlit.app`

## Local Installation

```bash
# Clone the repository
git clone https://github.com/matthijskorevaar/portfolio-theory-tool.git
cd portfolio-theory-tool

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run portfolio_app.py
```

## Usage for Teaching

This tool is designed for teaching finance courses, particularly:
- Portfolio Theory
- Mean-Variance Optimization
- CAPM and Asset Pricing
- Risk and Return Trade-offs

### Suggested Exercises

1. **Diversification**: Start with high correlation, then reduce it. Watch the frontier expand.

2. **News Shocks**: Apply positive news to one stock and ask students:
   - How does the frontier shift?
   - How does the market portfolio change?
   - What happens to the SML?

3. **Alpha Detection**: Show a stock above/below the SML and discuss mispricing.

4. **Risk Aversion**: Change the utility curve parameter to show different optimal portfolios.

## Course Context

Developed for Finance 1, a second-year bachelor course at Erasmus School of Economics.

## License

MIT License - Feel free to use and modify for educational purposes.
