import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class PricingParams:
    beta: float = 0.05 #elasticity
    gamma: float = 0.8 #time urgency
    q_max: float = 100.0 # init inv cap
    p_min: float = 10.0 #min price bound
    p_max: float = 100.0 #max price bound
    base_demand: float = 100.0 # base demand scaling factor

class DynamicPricingEngine:
    def __init__(self, params: PricingParams):
        self.params = params
        self.optimization_history = []

    def demand_function(self, p: float, t: float, q: float) -> float:
        price_elasticity = np.exp(-self.params.beta * p)
        time_urgency = 1 +self.params.gamma * t
        inv_constraint = min(1.0, q / self.params.q_max)

        demand = (self.params.base_demand * price_elasticity * time_urgency * inv_constraint)
        return demand
    
    def revenue_function(self, p: float, t:float, q: float) -> float:
        demand = self.demand_function(p, t, q)
        revenue = p * demand
        return revenue
    
    def negative_revenue(self, p: float, t: float, q: float) -> float:
        return -self.revenue_function(p,t,q)
    
    def compute_gradients(self, p: float, t: float, q: float) -> Tuple[float, float, float]:
        exp_term = np.exp(-self.params.beta *p)
        time_term = 1 + self.params.gamma *t
        dR_dp = (q *exp_term *time_term * (1 -self.params.beta * p))
        dR_dt = p * q *exp_term * self.params.gamma
        dR_dq = p * exp_term * time_term

        return dR_dp, dR_dt, dR_dq
    
    def compute_hessian(self, p: float, t: float, q: float) -> np.ndarray:
        exp_term = np.exp(-self.params.beta *p)
        time_term = 1 + self.params.gamma *t
        beta = self.params.beta
        gamma = self.params.gamma

        #second derivatives
        d2R_dp2 = (q * exp_term * time_term * (-2*beta + beta**2 * p))
        d2R_dt2 = 0.0
        d2R_dq2 = 0.0

        d2R_dpdt = q * exp_term * gamma * (1 - beta*p)
        d2R_dpdq = exp_term * time_term * (1 - beta*p)
        d2R_dtdq = p *exp_term * gamma

        hessian = np.array([[d2R_dp2, d2R_dpdt, d2R_dpdq],
                            [d2R_dpdt, d2R_dt2, d2R_dtdq],
                            [d2R_dpdq, d2R_dtdq, d2R_dq2]])
        
        return hessian
    
    def optimize_price(self, t: float, q:float, method: str = 'L-BFGS-B') -> Tuple[float, float]:
        if q <= 0:
            return self.params.p_min, 0.0
        
        def objective(p):
            return -self.revenue_function(p[0],t,q)
        
        if method == 'L-BFGS-B':
            p0 = np.array([1.0 / self.params.beta])
            p0 = np.clip(p0, self.params.p_min, self.params.p_max)

            bounds = [(self.params.p_min, self.params.p_max)]

            result = minimize(objective, p0, bounds=bounds, method='L-BFGS-B')
            optimal_price = result.x[0]
            max_revenue = -result.fun

        elif method == 'differential_evolution':
            #global optimization method
            bounds = [(self.params.p_min, self.params.p_max)]
            result = differential_evolution(objective, bounds, seed=42)
            optimal_price = result.x[0]
            max_revenue = -result.fun
        else:
            raise ValueError(f"unknown method: {method}")
        
        self.optimization_history.append({
            'time_remaining': t,
            'inventory': q,
            'optimal_price': optimal_price,
            'max_revenue': max_revenue
        })
        return optimal_price, max_revenue
    
    def gradient_descent_optimize(self, t: float, q: float, learning_rate: float = 0.1, iterations: int = 100, tolerance: float = 1e-6) -> Tuple[float, float]:
        p = (self.params.p_min + self.params.p_max) / 2 # start halfway through
        for iteration in range(iterations):
            if q <= 0:
                return self.params.p_min, 0.0
            
            dR_dp, _, _ = self.compute_gradients(p,t,q)
            #gradient ascent (maximize this)
            p_new = p +learning_rate * dR_dp
            #enforce bounds
            p_new = np.clip(p_new, self.params.p_min, self.params.p_max)

            #check convergence
            if abs(p_new - p) < tolerance:
                p = p_new
                break
            p=p_new

        revenue = self.revenue_function(p,t,q)
        return p, revenue

"""
Marketplace Simulation
"""

@dataclass
class CustomerAgent:
    #represent customer w price elasticicty
    elasticity: float #price sens
    value_multiplier: float #how much customers values the product
    willingness_to_pay: float # max price willing to pay

    def purchase_decision(self, current_price:float) ->bool:
        utility_diff = self.elasticity * (current_price - self.willingness_to_pay)
        purchase_prob = 1.0 / (1.0 + np.exp(utility_diff))
        return np.random.random() < purchase_prob
    
"""
Simulates 30 days of dynamic pricing marketplace
"""
class MarketplaceSimulator:
    def __init__(self, engine: DynamicPricingEngine, num_days: int = 30):
        self.engine = engine
        self.num_days = num_days
        self.daily_logs = []
        self.price_history =[]
        self.inventory_history = []
        self.revenue_history = []
        self.demand_history = []

    def generate_daily_customers(self, num_customers: int = 50) -> List[CustomerAgent]:
        customers = []
        for _ in range(num_customers):
            elasticity = np.random.normal(0.15, 0.08)
            elasticity = max(0.01, elasticity) #positive only
            wtp = np.random.uniform(30, 90)
            value_mult = np.random.uniform(0.8, 1.2)

            customers.append(CustomerAgent(
                elasticity=elasticity,
                value_multiplier=value_mult,
                willingness_to_pay=wtp
            ))
        return customers
    
    def run_simulation(self, verbose: bool = True) -> pd.DataFrame:
        curr_inv = self.engine.params.q_max
        total_revenue = 0.0

        if verbose:
            print("="*80)
            print("DYNAMIC PRICING ENGINE - 30 DAY MARKETPLACE SIMULATION")
            print("="*80)
            print(f"Initial Inventory: {curr_inv}")
            print(f"Price Bounds: ${self.engine.params.p_min:.2f} - ${self.engine.params.p_max:.2f}")
            print(f"Beta (elasticity): {self.engine.params.beta}")
            print(f"Gamma (urgency): {self.engine.params.gamma}")
            print("="*80)

        for day in range(1, self.num_days + 1):
            # Calculate normalized time remaining [0, 1]
            time_remaining = (self.num_days - day) / self.num_days

            # Optimize price for current state
            optimal_price, expected_revenue = self.engine.optimize_price(
                t=time_remaining,
                q=curr_inv,
                method='L-BFGS-B'
            )

            # Expected demand at optimal price
            expected_demand = self.engine.demand_function(
                optimal_price, time_remaining, curr_inv
            )

            # Generate daily customers
            daily_customers = self.generate_daily_customers(num_customers=50)

            # Process customer purchases
            daily_sales = 0
            daily_revenue = 0.0

            for customer in daily_customers:
                if curr_inv <= 0:
                    break

                if customer.purchase_decision(optimal_price):
                    daily_sales += 1
                    daily_revenue += optimal_price
                    curr_inv -= 1

            total_revenue += daily_revenue

            daily_log = {
                'day': day,
                'time_remaining': time_remaining,
                'inventory_start': self.engine.params.q_max - (self.engine.params.q_max - curr_inv),
                'inventory_end': curr_inv,
                'optimal_price': optimal_price,
                'expected_demand': expected_demand,
                'actual_sales': daily_sales,
                'daily_revenue': daily_revenue,
                'cumulative_revenue': total_revenue,
                'utilization': daily_sales / len(daily_customers) if daily_customers else 0
            }
            
            self.daily_logs.append(daily_log)
            self.price_history.append(optimal_price)
            self.inventory_history.append(curr_inv)
            self.revenue_history.append(daily_revenue)
            self.demand_history.append(expected_demand)

            if verbose and (day % 5 == 0 or day == 1 or day == self.num_days):
                print(f"Day {day:2d} | Time: {time_remaining:.2%} | Price: ${optimal_price:6.2f} | "
                      f"Inventory: {curr_inv:5.0f} | Sales: {daily_sales:3d} | "
                      f"Daily Rev: ${daily_revenue:7.2f} | Total Rev: ${total_revenue:8.2f}")
                
        if verbose:
            print("="*80)
            print(f"FINAL RESULTS")
            print("="*80)
            print(f"Total Units Sold: {int(self.engine.params.q_max - curr_inv)}")
            print(f"Remaining Inventory: {curr_inv:.0f}")
            print(f"Total Revenue: ${total_revenue:.2f}")
            print(f"Average Price: ${np.mean(self.price_history):.2f}")
            print(f"Average Daily Sales: {np.mean([log['actual_sales'] for log in self.daily_logs]):.1f}")
            print("="*80)

        return pd.DataFrame(self.daily_logs)
        
    def plot_results(self, figsize: Tuple[int, int] = (16,10)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Dynamic Pricing Engine - 30 Day Simulation Results', fontsize=16, fontweight='bold')
        
        days = range(1, self.num_days + 1)
        
        # Plot 1: Price Evolution
        ax = axes[0, 0]
        ax.plot(days, self.price_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=1/self.engine.params.beta, color='r', linestyle='--', 
               label=f'Theoretical Optimum: ${1/self.engine.params.beta:.2f}')
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_title('Optimal Price Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Inventory Depletion
        ax = axes[0, 1]
        ax.plot(days, self.inventory_history, 'g-', linewidth=2, marker='s', markersize=4)
        ax.fill_between(days, self.inventory_history, alpha=0.3, color='green')
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Inventory (units)', fontsize=11)
        ax.set_title('Inventory Depletion Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Daily Revenue
        ax = axes[1, 0]
        ax.bar(days, self.revenue_history, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Daily Revenue ($)', fontsize=11)
        ax.set_title('Daily Revenue Generation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Price vs Inventory Relationship
        ax = axes[1, 1]
        scatter = ax.scatter(self.inventory_history, self.price_history, 
                        c=days, cmap='viridis', s=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Inventory (units)', fontsize=11)
        ax.set_ylabel('Price ($)', fontsize=11)
        ax.set_title('Price Optimization vs Inventory Level', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Day', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
                
    def plot_revenue_surface(self, figsize: Tuple[int, int] = (12, 5)):
        #create grid of prices and inventories
        t_fixed = 0.5
        prices = np.linspace(self.engine.params.p_min, self.engine.params.p_max, 50)
        inventories = np.linspace(1, self.engine.params.q_max, 50)

        prices_grid, inventories_grid = np.meshgrid(prices,inventories)
        revenue_grid = np.zeros_like(prices_grid)

        for i in range(len(prices)):
            for j in range(len(inventories)):
                revenue_grid[j,i] = self.engine.revenue_function(prices_grid[j,i], t_fixed, inventories_grid[j,i])
        fig, axes = plt.subplots(1,2,figsize=figsize)
        fig.suptitle(f'Revenue Surface (t={t_fixed:.1%} time remaining)', fontsize = 14, fontweight = 'bold')
            
        #3d surface plot
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(prices_grid, inventories_grid, revenue_grid, cmap='viridis', alpha = 0.8)
        ax.set_xlabel('Price ($)', fontsize=10)
        ax.set_ylabel('Inventory (units)', fontsize=10)
        ax.set_zlabel('Revenue ($)', fontsize=10)
        ax.set_title('3D Revenue Surface', fontsize=11, fontweight='bold')

        #contour plot
        ax = axes[1]
        contour = ax.contourf(prices_grid, inventories_grid, revenue_grid, levels=20, cmap='viridis')
        ax.contour(prices_grid, inventories_grid, revenue_grid, levels = 10, colors = 'black', alpha= 0.3, linewidths=0.5)

        #plot optimal price points over time
        for day, log in enumerate(self.daily_logs):
            ax.plot(log['optimal_price'], log['inventory_start'], 'r*', markersize=10)
            
        ax.set_xlabel('Price ($)', fontsize=10)
        ax.set_ylabel('Inventory (units)', fontsize=10)
        ax.set_title('Contour Plot (Red stars = daily optima)', fontsize=11, fontweight='bold')
        plt.colorbar(contour, ax=ax, label='Revenue ($)')
        
        plt.tight_layout()
        return fig
        
"""
MAIN EXECUTION
"""
if __name__ == "__main__":
    params = PricingParams(
        beta = 0.05,        #Price Elasticity
        gamma = 0.8,        #Time Urgency
        q_max = 100.0,      #Initial Inventory
        p_min = 10.0,       #Min Price
        p_max = 100.0,      #Max Price
        base_demand = 100.0 #Base Demand Scaling
    )

    engine = DynamicPricingEngine(params)

    #Run marketplace simulation
    simulator = MarketplaceSimulator(engine, num_days=30)
    results_df = simulator.run_simulation(verbose=True)

    #Save results to csv
    results_df.to_csv('dynamic_pricing_results.csv', index = False)
    print("\n Results saved to 'dynamic_pricing_results.csv'")

    #Generate visualizations
    print("\nGenerating visualizations")
    fig1 = simulator.plot_results()
    fig1.savefig('pricing_simulation_results.png', dpi=300, bbox_inches='tight')
    print("Saved 'pricing_simulation_results.png'")

    fig2 = simulator.plot_revenue_surface()
    fig2.savefig('revenue_surface.png', dpi=300, bbox_inches='tight')
    print("Saved 'revenue_surface.png'")
    
    plt.show()

    #Print summary stats
    print("\n" + "="*80)
    print("SUMMARY STATS")
    print("="*80)
    print(results_df.describe())
    print("\n" + "="*80)