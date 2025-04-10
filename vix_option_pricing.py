import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import logging

from numpy.polynomial.legendre import Legendre, leggauss
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import iv  # Bessel I_\nu
from scipy.optimize import minimize_scalar, minimize

# =============================================
#      1. Read & Fit Historical VIX Data (Legendre Part)
# =============================================

def prepare_legendre_calibration():
    """
    Read historical VIX data and perform polynomial fitting to prepare for Legendre model calibration.
    """
    excel_path_vix = "./VIXdata_from_1990_01_02.xlsx"
    data_hist = pd.read_excel(excel_path_vix, header=None)
    vix_data = data_hist[0].values
    vix_data = vix_data[~np.isnan(vix_data)]
    vix_data = vix_data / 100.0  # Scale to [0,1]

    # Sort & Construct Empirical CDF
    vix_sorted = np.sort(vix_data)
    N_hist = len(vix_sorted)
    cdf_values = np.linspace(1/N_hist, 1, N_hist)

    h_interp = interp1d(
        cdf_values,
        vix_sorted,
        kind='linear',
        bounds_error=False,
        fill_value=(vix_sorted[0], vix_sorted[-1])
    )

    def h(u):
        """F^{-1}(u),  u ∈ [0,1]"""
        return h_interp(u)

    # Polynomial fitting of h(u)
    deg = 30
    u_train = np.linspace(0,1,1001)
    h_train = h(u_train)

    coeffs = np.polyfit(u_train, h_train, deg)  # polyfit returns coefficients arranged from highest degree to lowest
    h_poly = np.poly1d(coeffs)

    def h_poly_eval(u):
        return h_poly(u)

    def tilde_h(x):
        """
        tilde_h(x) = h( (x+1)/2 ),  x ∈ [-1,1].
        """
        return h_poly_eval((x+1)/2)

    def tilde_h1(x, K):
        """ payoff = (tilde_h(x) - K)^+ """
        return np.maximum(tilde_h(x) - K, 0.0)

    # Legendre-Gauss nodes
    xg, wg = leggauss(200)

    def compute_inner_products_h1(K):
        """
        Compute <tilde_h1, P_n> for all n=0..deg, return an array of length (deg+1)
        where tilde_h1(x) = (tilde_h(x) - K)^+, P_n = Legendre.basis(n)(x).
        """
        inner_products = []
        for n in range(deg+1):
            Pn_vals = Legendre.basis(n)(xg)
            integrand = tilde_h1(xg, K) * Pn_vals
            val = np.sum(integrand * wg)
            inner_products.append(val)
        return np.array(inner_products)

    return {
        "tilde_h": tilde_h,
        "tilde_h1": tilde_h1,
        "compute_inner_products_h1": compute_inner_products_h1,
        "deg": deg,
        "xg": xg,
        "wg": wg
    }

# =============================================
#     2. 3/2 Model Function Definitions (Pricing + Calibration)
# =============================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
def price_vix_call_3_2(V, K, T, alpha, beta, kappa, r):
    """
    Goard & Mazur (2013) 3/2 model: VIX call pricing formula
    """
    if T <= 1e-10:
        return max(V - K, 0.0)

    p = 1.0 - np.exp(-alpha*T)
    nu = 1.0 - 2.0*beta/(kappa**2)

    # Ensure nu is positive to avoid issues with Bessel functions
    if nu <= 0:
        return 1e10  # Large penalty

    try:
        exponent_term = -2.0*alpha*np.exp(-alpha*T) / (kappa**2 * V * p)
        if np.isinf(exponent_term) or np.isnan(exponent_term):
            return 1e10  # Large penalty

        factor_out = (
            (2.0*alpha*np.exp(-r*T)) / (kappa**2*p)
            * np.exp(exponent_term)
            * (V**(-beta/(kappa**2)+0.5))
            * np.exp(alpha*T*(-beta/(kappa**2)+0.5))
        )
    except OverflowError:
        return 1e10  # Large penalty

    X = 1.0/K

    def integrand(u):
        try:
            z = (4.0*alpha*np.sqrt(u)*np.exp(-0.5*alpha*T)) / (kappa**2*np.sqrt(V)*p)
            if z <= 0:
                return 0.0
            term1 = u**(0.5 - beta/(kappa**2))
            term2 = (1.0/u - K)
            term3 = np.exp(-2.0*alpha*u / (kappa**2*p))
            bessel = iv(nu, z)
            if np.isnan(bessel) or np.isinf(bessel):
                return 0.0
            return term1 * term2 * term3 * bessel
        except:
            return 0.0

    try:
        # Start integration from a small positive number to avoid u=0
        integral_val, _ = quad(integrand, 1e-6, X, limit=200, epsabs=1e-10, epsrel=1e-10)
        if np.isnan(integral_val) or np.isinf(integral_val):
            return 1e10  # Large penalty
        call_price = factor_out * integral_val
        return max(call_price, 0.0)
    except Exception as e:
        # Log the exception if needed
        logging.debug(f"Integration failed with error: {e}")
        return 1e10  # Large penalty

def calibrate_3_2_model(df_, K, r):
    """
    Calibrate 3/2 model parameters after removing NaN values.
    """
    # Remove rows with NaN in relevant columns
    df_clean = df_.dropna(subset=["VIX index", "Time to expiry", "VIX call option price"])

    V_vals = df_clean["VIX index"].values
    T_vals = df_clean["Time to expiry"].values
    C_mkt  = df_clean["VIX call option price"].values

    def objective(params):
        alpha_, beta_, kappa_ = params
        sse = 0.0
        for i in range(len(df_clean)):
            model_price = price_vix_call_3_2(
                V=V_vals[i],
                K=K,
                T=T_vals[i],
                alpha=alpha_,
                beta=beta_,
                kappa=kappa_,
                r=r
            )
            if model_price >= 1e10:
                return 1e20  # Very large penalty to discourage this region
            sse += (model_price - C_mkt[i])**2
        return sse

    # Improved initial guesses based on typical parameter ranges
    x0 = [0.5, -2.0, 1.0]  # (alpha, beta, kappa)
    bnds = [
        (0.1, 2.0),      # alpha
        (-10.0, -0.5),   # beta
        (0.5, 3.0)        # kappa
    ]

    # Suppress warnings during optimization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(objective, x0, bounds=bnds, method='L-BFGS-B', options={'maxiter': 500})

    if res.success and res.fun < 1e20:
        alpha_hat, beta_hat, kappa_hat = res.x
        logging.info(f"[3/2] Calibration succeeded with alpha={alpha_hat:.4f}, beta={beta_hat:.4f}, kappa={kappa_hat:.4f}")
    else:
        alpha_hat, beta_hat, kappa_hat = x0
        logging.warning("[3/2] Calibration failed, using fallback parameters.")
    return alpha_hat, beta_hat, kappa_hat

# =============================================
#     3. Legendre Model Pricing
# =============================================

def calibrate_legendre_model(df, K, r, calibration_data, deg=30):
    """
    Apply Legendre model calibration and pricing to a single dataset.
    """
    compute_inner_products_h1 = calibration_data["compute_inner_products_h1"]
    deg = calibration_data["deg"]
    xg = calibration_data["xg"]
    wg = calibration_data["wg"]

    # (a) First compute <tilde{h}_1, P_n> for this K
    inner_products_h1 = compute_inner_products_h1(K)

    # (b) Construct Pn(x_i)
    T_t_i = df['Time to expiry'].values
    x_i   = df['x value'].values
    c_i   = df['VIX call option price'].values
    N_i     = len(df)

    Pn_matrix = np.zeros((N_i, deg+1))
    for n in range(deg+1):
        Pn_matrix[:, n] = Legendre.basis(n)(x_i)

    tilde_nu = np.zeros((N_i, deg+1))
    for n in range(deg+1):
        factor = (2*n+1)/2.0 * inner_products_h1[n]
        tilde_nu[:, n] = factor * Pn_matrix[:, n]

    n_array = np.arange(deg+1)

    def objective_legendre(k):
        error = 0.0
        for i in range(N_i):
            exponent = np.exp(-0.5*k*n_array*(n_array+1)* T_t_i[i])
            model_price = np.exp(-r*T_t_i[i]) * np.sum(tilde_nu[i,:] * exponent)
            error += (model_price - c_i[i])**2
        return error

    res_legendre = minimize_scalar(objective_legendre, bounds=(4, 7), method='bounded')
    if res_legendre.success:
        hat_k = res_legendre.x
        print(f"[Legendre] Calibrated k from full data = {hat_k}")
    else:
        hat_k = 0.02
        print("[Legendre] Optimization failed, use fallback = 0.02")

    # Calculate Legendre model prices
    legendre_prices = []
    for i in range(N_i):
        exponent = np.exp(-0.5*hat_k*n_array*(n_array+1)*T_t_i[i])
        summation = np.sum(tilde_nu[i,:] * exponent)
        model_price_legendre = np.exp(-r*T_t_i[i]) * summation
        legendre_prices.append(model_price_legendre)

    return hat_k, legendre_prices

# =============================================
#     4. Function to Process a Single File
# =============================================

def process_single_file(file_path, calibration_data, K=0.20, r=0.0376, output_dir="output"):
    """
    Process a single option contract file, perform model calibration and pricing, and save the results.
    """
    # Read data
    df = pd.read_excel(file_path)
    
    # Replace inf and -inf with nan without using inplace=True
    if '3/2 Model Price' in df.columns:
        df['3/2 Model Price'] = df['3/2 Model Price'].replace([np.inf, -np.inf], np.nan)
    else:
        df['3/2 Model Price'] = np.nan  # Initialize with NaN if the column doesn't exist

    # Calibrate Legendre model
    hat_k, legendre_prices = calibrate_legendre_model(df, K, r, calibration_data)
    df['Legendre Price'] = legendre_prices

    # Calibrate 3/2 model
    alpha_hat, beta_hat, kappa_hat = calibrate_3_2_model(df, K, r)
    print("3/2 model params:")
    print("alpha =", alpha_hat, "beta =", beta_hat, "kappa =", kappa_hat)

    # 3/2 model pricing
    three_half_prices = []
    for i in range(len(df)):
        V_ = df.loc[i, 'VIX index']
        T_ = df.loc[i, 'Time to expiry']
        model_price_3_2 = price_vix_call_3_2(V_, K, T_, alpha_hat, beta_hat, kappa_hat, r)
        three_half_prices.append(model_price_3_2)
    df['3/2 Model Price'] = three_half_prices

    # Calculate errors
    df['Abs Error (Legendre)'] = np.abs(df['VIX call option price'] - df['Legendre Price'])
    df['Rel Error (Legendre)'] = df['Abs Error (Legendre)'] / df['VIX call option price']

    df['Abs Error (3/2)'] = np.abs(df['VIX call option price'] - df['3/2 Model Price'])
    df['Rel Error (3/2)'] = df['Abs Error (3/2)'] / df['VIX call option price']

    # Output error statistics
    avg_abs_error_legendre = df['Abs Error (Legendre)'].mean()
    avg_rel_error_legendre = df['Rel Error (Legendre)'].mean()
    avg_abs_error_32 = df['Abs Error (3/2)'].mean()
    avg_rel_error_32 = df['Rel Error (3/2)'].mean()

    print("\n========== Results Comparison ==========")
    print(df[[
        'Date',
        'VIX index',
        'VIX call option price',
        'Legendre Price',
        '3/2 Model Price',
        'Abs Error (Legendre)',
        'Rel Error (Legendre)',
        'Abs Error (3/2)',
        'Rel Error (3/2)',
        'Time to expiry'
    ]])

    print("\n========== Average Error Comparison ==========")
    print(f"Average Absolute Error (Legendre): {avg_abs_error_legendre:.6f}")
    print(f"Average Relative Error (Legendre): {avg_rel_error_legendre:.2%}")
    print(f"Average Absolute Error (3/2 Model): {avg_abs_error_32:.6f}")
    print(f"Average Relative Error (3/2 Model): {avg_rel_error_32:.2%}")

    # Filter out NaN values in 3/2 Model Price
    mask_valid_3_2 = df['3/2 Model Price'].notna()
    df_filtered = df[mask_valid_3_2].copy()
    df_filtered.reset_index(drop=True, inplace=True)

    # Recalculate errors after filtering
    avg_abs_error_32_filtered = df_filtered['Abs Error (3/2)'].mean()
    avg_rel_error_32_filtered = df_filtered['Rel Error (3/2)'].mean()

    print("\n========== After Filtering out 3/2 Model's Inf/NaN Rows ==========")
    print(f"Number of filtered rows: {len(df) - len(df_filtered)}")
    print(f"Remaining rows: {len(df_filtered)}")

    print("\n=== Average Error Comparison (3/2 Model, Filtered) ===")
    print(f"Average Absolute Error (3/2 Model): {avg_abs_error_32_filtered:.6f}")
    print(f"Average Relative Error (3/2 Model): {avg_rel_error_32_filtered:.2%}")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct output file name
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, f"results_{file_name}")

    # Save results to Excel
    df.to_excel(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(df['Date'], df['VIX call option price'], label='Market Price', marker='o')
    plt.plot(df['Date'], df['Legendre Price'], label='Legendre Price', marker='x')
    plt.plot(df['Date'], df['3/2 Model Price'], label='3/2 Model Price', marker='s')

    plt.xlabel('Date')
    plt.ylabel('VIX Call Option Price')
    plt.title(f'Market vs. Legendre vs. 3/2 ({file_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, f"plot_{file_name.replace('.xlsx', '.png')}")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}\n")

# =============================================
#     5. Main Program
# =============================================

def main():
    # Prepare data required for Legendre calibration
    calibration_data = prepare_legendre_calibration()

    # Paths to four option contract files
    file_paths = [
        "./20241030/20241030_call_20.xlsx",
        "./20241120/20241120_call_20.xlsx",
        "./20241218/20241218_call_20.xlsx",
        "./20241224/20241224_call_20.xlsx"
    ]

    # Process each file
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        process_single_file(file_path, calibration_data)
        print("="*80)

if __name__ == "__main__":
    main()
