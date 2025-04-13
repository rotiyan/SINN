import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class FXForwardExposureSimulator:
    def __init__(self, S0=0.91, r_usd=0.06, r_eur=0.05, sigma=0.10,
                 T=5.0, horizon=5.0, steps=12, n_paths=1000,
                 regulatory_multiplier=1.4, notional=1e6, seed=42):
        """
        Initialize the simulator with market parameters and simulation settings.

        Parameters:
        - S0: Initial FX spot rate (EUR per USD)
        - r_usd: USD interest rate (base currency)
        - r_eur: EUR interest rate (term currency)
        - sigma: Annualized FX volatility
        - T: Maturity of the FX forward (in years)
        - horizon: Simulation horizon (in years)
        - steps: Number of time steps in the simulation horizon
        - n_paths: Number of Monte Carlo simulation paths
        - regulatory_multiplier: Multiplier for EAD calculation (e.g., 1.4)
        - notional: The notional of the fxforward.
        - seed: Random seed for reproducibility
        """
        self.S0 = S0
        self.r_usd = r_usd
        self.r_eur = r_eur
        self.sigma = sigma
        self.T = T
        self.horizon = horizon
        self.steps = steps
        self.dt = horizon / steps
        self.n_paths = n_paths
        self.reg_multiplier = regulatory_multiplier
        self.notional = notional
        self.seed = seed

        #torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def simulate_paths(self):
        """
        Simulate FX spot rate paths (EUR per USD) using Geometric Brownian Motion under
        the domestic (USD) risk-neutral measure.
        """
        Z = torch.randn((self.n_paths, self.steps), device=self.device)
        drift = (self.r_usd - self.r_eur - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * torch.sqrt(torch.tensor(self.dt)) * Z
        increments = drift + diffusion
        log_paths = torch.cumsum(torch.cat([
            torch.zeros((self.n_paths, 1), device=self.device), increments
        ], dim=1), dim=1)
        self.S_paths = self.S0 * torch.exp(log_paths)

    def compute_mtm(self):
        """
        Compute mark-to-market (MTM) values of the FX forward contract
        for each simulation path at each time step.
        """
        time_grid = torch.linspace(0, self.horizon, self.steps + 1, device=self.device)
        F0 = self.S0 * torch.exp( torch.tensor((self.r_usd - self.r_eur) * self.T) )
        MTM = torch.zeros_like(self.S_paths)

        for i, t in enumerate(time_grid):
            tau = self.T - t.item()
            if tau > 0:
                F_t_T = self.S_paths[:, i] * torch.exp( torch.tensor((self.r_usd - self.r_eur) * tau))
                MTM[:, i] = (F_t_T - F0)*self.notional
        self.MTM = MTM
        self.time_grid = time_grid.cpu().numpy()

    def compute_exposure(self):
        """
        Calculate Expected Positive Exposure (EPE) and Exposure at Default (EAD)
        from the simulated MTM values.
        """
        pos_MTM = torch.maximum(self.MTM, torch.tensor(0.0, device=self.device))
        EPE = pos_MTM.mean(dim=0).cpu().numpy()
        EAD = EPE * self.reg_multiplier
        self.results_df = pd.DataFrame({
            'Time (years)': self.time_grid,
            'EPE': EPE,
            'EAD': EAD
        })

    def plot_mtm_paths(self,MTM, n_plot=50):
        """
        Plot a subset of simulated MTM paths to visualize exposure dynamics.

        Parameters:
        - n_plot: Number of paths to plot (default is 50)
        """
        MTM_sample = MTM[:n_plot].cpu().numpy()
        plt.figure(figsize=(12, 6))
        for i in range(n_plot):
            plt.plot(self.time_grid, MTM_sample[i], alpha=0.3)
        plt.title("Simulated MTM Paths for FX Forward (EUR per USD)")
        plt.xlabel("Time (years)")
        plt.ylabel("MTM (USD)")
        plt.grid(True)
        plt.show()

    def compute_autocorrelation(self, MTM, lags=None):
        """
        Compute the autocorrelation function of MTM paths averaged over all simulations.

        Returns:
        - autocorr: Autocorrelation values as a NumPy array
        """
        mtm_centered = MTM - MTM.mean(dim=1, keepdim=True)
        if lags is None:
            lags = torch.arange(mtm_centered.shape[0])
        elif isinstance(lags,int):
            lags = torch.arange(lags)
        else:
            lags = torch.tensor(lags, dtype=torch.int32)

            
        corr = torch.zeros((len(lags), *mtm_centered.shape[2:]),device=mtm_centered.device)
        for i, lag in enumerate(lags):
            if lag ==0:
                u=v=mtm_centered
            elif lag < mtm_centered.shape[0]:
                u,v = mtm_centered[:-lag, ...], mtm_centered[lag:, ...]
            else:
                continue
            corr[i, ...] = torch.sum(u*v, axis=[0,1])/(
                torch.sqrt(
                    torch.sum(torch.square(u), axis=[0,1])*
                    torch.sum(torch.square(v), axis=[0,1])
                )
            )
        return corr

    def compute_pdf(self, MTM, lower, upper, n, bw=0.5):
        """
        Estimate the PDF using a Gaussian KDE implemented in PyTorch.

        Parameters:
        - time_index: Index in the time grid to evaluate the MTM distribution
        - bandwidth: Bandwidth for the KDE

        Returns:
        - x_vals: X-axis values for plotting
        - kde_vals: Estimated PDF values
        """
        x = torch.ravel(MTM)
        grid = torch.linspace(lower, upper, n, device=x.device)
        norm_factor = 2*torch.pi**0.5*len(x)*bw

        return torch.sum(torch.exp( -0.5*torch.square((x[:,None] - grid[None,:]) /bw)),axis=0)/norm_factor
        

    def run(self):
        """
        Run the full simulation process: generate FX paths, compute MTM values,
        and derive EPE and EAD.

        Returns:
        - results_df: A pandas DataFrame containing time, EPE, and EAD
        """
        self.simulate_paths()
        self.compute_mtm()
        self.compute_exposure()
        return self.results_df

    def get_simpaths(self):
        return self.MTM
