#Wrappers
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

#write a CONVENIENCE FUNCTION for reading Portfolios_Formed_on_ME_monthly_EW.csv
def get_ffme_returns():
    """
    Load and format the Fama-Frech Dataset for the returns of the Top and Bottom Decilces by MarketCap
    """

    rets = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                      header=0, #set the first row as column index
                      index_col=0, parse_dates=True, #set the first column as row index and tell pandas to convert number to date
                      na_values=-99.99 #tell pandas that missing values have been encoded as -99.99
    )

    rets = rets[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m")
    rets.index = rets.index.to_period('M')
    return rets

#write a CONVENIENCE FUNCTION for reading edhec-hedgefundindices.csv
def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


#write a CONVENIENCE FUNCTION for reading ind30_m_vw_rets.csv
def get_ind_returns():
    """
    Load and format the Fama-Frech Dataset for the returns of the Top and Bottom Decilces by MarketCap
    """
    ind = pd.read_csv("ind30_m_vw_rets.csv", header=0, index_col=0, parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

#write a CONVENIENCE FUNCTION for reading ind30_m_size.csv
def get_ind_size():
    """
    Load and format ind30_m_size.csv from working directory
    """
    ind = pd.read_csv("ind30_m_size.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

#write a CONVENIENCE FUNCTION for reading ind30_m_nfirms.csv
def get_ind_nfirms():
    """
    Load and format ind30_m_nfirms.csv from working directory
    """
    ind = pd.read_csv("ind30_m_nfirms.csv", header=0, index_col=0, parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

#write a CONVENIENCE FUNCTION for obtaining total market index returns from ind30
def get_total_market_index_returns():
    ret = get_ind_returns()
    size = get_ind_size()
    n = get_ind_nfirms()
    ind_mktcap = n * size
    total_mktcap = ind_mktcap.sum(axis="columns")
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight*ret).sum(axis="columns")
    return total_market_return

#calculate drawdown
def draw(ReturnSeries: pd.Series):
    
    #Create a documentation string for the function
    """
    Takes a time sereis of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdown
    """

    WealthIndex = 1000*(1+ReturnSeries).cumprod() #assumes starting value = $1000
    PreviousPeaks = WealthIndex.cummax()
    Drawdowns = (WealthIndex - PreviousPeaks) / PreviousPeaks

    return pd.DataFrame({
        "Wealth": WealthIndex,
        "Peaks": PreviousPeaks,
        "Drawdowns": Drawdowns
    })

#calculate skewness
def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """

    #"De-meaned r" aka "r centered about the mean"
    demean_r = r - r.mean()
    sigma_r = r.std(ddof=0) #uses the population standard deviation by setting Degrees of Freedom equal to zero: ddof=0
    exp = (demean_r**3).mean()
    return exp/sigma_r**3

#calculate kurtosis
def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """

    #"De-meaned r" aka "r centered about the mean"
    demean_r = r - r.mean()
    sigma_r = r.std(ddof=0) #uses the population standard deviation by setting Degrees of Freedom equal to zero: ddof=0
    exp = (demean_r**4).mean()
    return exp/sigma_r**4

#tells us if returns are normally-distributed
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normally distributed
    Test is applied at the 1% level by default
    Returns True if the null hypothesis of normality holds, false if the null hypothesis is rejected
    """
    import scipy.stats
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

#calculate semi-deviation
def semideviation3(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation

#VaR historical
def var_historic(r, level=5):
    """
    VaR Historic
    Returns the 5th percentile by default
    """
    import numpy as np
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("r must be of type Series or DataFrame")
    
#VaR calculated via Gaussian method or Modified Gaussian (cornish-fisher) method
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    z = norm.ppf(level/100)
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z**3 - 5*z)*(s**2)/36)

    return -(r.mean() + z*r.std(ddof=0))

#CVaR based on historical VaR
def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of a Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("r must be of type Series or DataFrame")
    
#CVaR based on Gaussian or Modified Gaussian (Cornish-Fisher) VaR
def cvar_gaussian(r, level=5, modified=False):
    """
    Computes the Conditional VaR of a Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_gaussian(r, level=level, modified=modified)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_gaussian, level=level, modified=modified)
    else:
        raise TypeError("r must be of type Series or DataFrame")

#calculate annualized returns
def annualize_ret(r, periods_per_year):
    """
    Returns the Compounded Annulized Growth Rate (CAGR) of a set of returns
    *Pre-Processing: Must set r row index equal to dates and column index equal to asset names*
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    CAGR = compounded_growth**(periods_per_year/n_periods)-1
    return CAGR

#calculate annualized volatility (annualized standard deviation)
def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """
    return r.std()*(periods_per_year**0.5)

#calculate Sharpe Ratio
def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    *Pre-Processing: Must set r row index equal to dates and column index equal to asset names*
    """
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_excess_ret = annualize_ret(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_excess_ret/ann_vol

#Calculate portfolio returns
def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    #Using matrix algebra:
    return weights.T @ returns

#Calculate portfolio volatility
def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    #Using matrix algebra:
    return (weights.T @ covmat @ weights)**0.5

#plot a 2-asset efficient frontier
def plot_ef2(n_points, er, cov):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only handle 2 assets")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    return ef.plot.line(x="Volatility", y="Returns", style=".-")

#plot a k-asset efficient frontier (next three functions)
def plot_ef(n_points, er, cov, min=0.0, max=1.0, optimizer="SLSQP", riskfree_rate=0, show_cml=True, show_ew=True, show_gmv=True):
    """
    Plots the k-asset efficient frontier
    defaults to a quadratic optimizer from scipy called "SLSQP"
    min/max weight per asset constraint defaults to min=0.0, max=1.0
    use show_cml=False to hide the Capital Market Line (set to True by default)
    use show_ew=False to hide the the naive portfolio (set to True by default)
    use show_gmv=False to hide the the naive portfolio (set to True by default)
    default risk-free rate =0
    """
    weights = optimal_weights(n_points, er, cov, min=min, max=max, optimizer=optimizer)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=".-")    
    if show_ew:
        k = er.shape[0]
        w_ew = np.repeat(1/k, k)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        #Display the equally-weighted portfolio (naive diversification)
        ax.plot([vol_ew], [r_ew], color="red", marker="o")
    if show_gmv:
        k = er.shape[0]
        w_gmv = gmv(cov, min=min, max=max, optimizer=optimizer)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        #Display the minimum-variance portfolio
        ax.plot([vol_gmv], [r_gmv], color="black", marker="o") 
    if show_cml:
        #Find tangency asset weights, portfolio return, and portfolio volatility
        w_msr = msr(riskfree_rate, er, cov, min=min, max=max, optimizer=optimizer)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        #Add the Capital Market Line (CML) to efficient frontier
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y,
                color="green",
                marker="o",
                linestyle="dashed")
    
    return ax

def gmv(cov, min=0.0, max=1.0, optimizer="SLSQP"):
    """
    Returns the weights for Global Minimum Volatility portfolio, given covariance matrix
    """
    k = cov.shape[0]
    return msr(0, np.repeat(1, k), cov, min=min, max=max, optimizer=optimizer)

def optimal_weights(n_points, er, cov, min=0.0, max=1.0, optimizer="SLSQP"):
    """
    Generates a list of weights to run the optimizer on
    defaults to a quadratic optimizer from scipy called "SLSQP"
    min/max weight per asset constraint defaults to min=0.0, max=1.0
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov, min=min, max=max, optimizer=optimizer) for target_return in target_rs]
    return weights

def minimize_vol(target_return, er, cov, min=0.0, max=1.0, optimizer="SLSQP"):
    """
    target_return -> W
    defaults to a quadratic optimizer from scipy called "SLSQP"
    min/max weight per asset constraint defaults to min=0.0, max=1.0
    """
    k = er.shape[0]
    initial_guess = np.repeat(1/k, k)
    bounds = ((min,max),)*k
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    results = minimize(portfolio_vol, initial_guess,
                       args=(cov,), method=optimizer,
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                       )
    return results.x
#Some notes about minimize_vol():
#   A child tuple of form ((x,y),)*k repeats (x,y) k number of times to form the parent tuple
#   type is equality
#   send er as an additional argument
#   lambda function aka annonymous function


#solve asset weights for maximum Sharpe Ratio
def msr(riskfree_rate, er, cov, min=0.0, max=1.0, optimizer="SLSQP"):
    """
    RiskFree, Expected Return, Covariance Matrix -> Tangent Portfolio Weights
    defaults to a quadratic optimizer from scipy called "SLSQP"
    min/max weight per asset constraint defaults to min=0.0, max=1.0
    """
    k = er.shape[0]
    initial_guess = np.repeat(1/k, k)
    bounds = ((min,max),)*k
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) -1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Multiplies the Sharpe Ratio by -1, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -1*(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, initial_guess,
                       args=(riskfree_rate, er, cov,), method=optimizer,
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                       )
    return results.x

#CPPI
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Executes a dynamic CPPI strategy, given a pandas Series or DataFrame containing the set of returns for the risky asset
    Returns a dictionary of DataFrames for: asset value history, risk budget history, risky asset weight history
    Must either specify a riskfree_rate assumption (defaults to 0.03) or safe_r (defaults to 'None')
    """
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor

    #Make it so the function can handle pd.Series by immediately converting it to a DataFrame
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns= ["R"])
    
    #Use the risk free rate if safe_r is not specified.
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12
    
    #Innitialize CPPI parameters
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    peak = start

    #Implement the CPPI algorithim with a for() loop
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1) #constraint: risky_w cannot exceed 100% i.e., we do not allow levered CPPI investment (must borrow to finance risky_w > 100%)
        risky_w = np.maximum(risky_w, 0) #constraint: cannot short the risky asset
        safe_w = 1 - risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        ## update the account value for this time step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        ## populate account_history, cushion_history, risky_w_history
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
    
    #Other calculations
    risky_wealth = start*(1+risky_r).cumprod()
    results = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return results

#Summary Statistics
def summary_stats(r, riskfree_rate=0.03, freq=12):
    """
    Return a DataFrame that contains aggregated summary statistics for returns
    Defaults to assuming a set of MONTHLY returns (freq=12)
    Defaults to an annualized risk free rate of 0.03
    """
    ann_r = r.aggregate(annualize_ret, periods_per_year=freq)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=freq)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=freq)
    dd = r.aggregate(lambda r: draw(r).Drawdowns.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Modified VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

## Geometric Brownian Motion (GBM)
def gbm(s_0, n_years, mu, sigma, steps, n_sims=1000):
    """
    Evolution of an initial stock price using a Geometric Brownian Motion
    """
    dt = 1/steps
    n_steps = int(n_years*steps)
    rets = np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_sims))
    rets[0] = 1
    prices = s_0*pd.DataFrame(rets).cumprod()
    return prices