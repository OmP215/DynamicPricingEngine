import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HotelParameterEstimator:

    """
    Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
    """
    def __init__(self, csv_path="hotel_bookings.csv"):
        print("Loading Hotel Bookings Dataset")
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()

    def _preprocess_data(self):
        print("Preprocesssing data...")
        self.df['arrival_date'] = pd.to_datetime(
            self.df['arrival_date_year'].astype(str) + '-' +
            self.df['arrival_date_month'].astype(str) + '-' +
            self.df['arrival_date_day_of_month'].astype(str),
            format = '%Y-%B-%d',
            errors = 'coerce'
        )

        if 'reservation_status_date' in self.df.columns:
            self.df['booking_date'] = pd.to_datetime(self.df['reservation_status_date'], errors = 'coerce')
        else:
            self.df['booking_date'] = self.df['arrival_date'] - pd.to_timedelta(self.df['lead_time'], unit='D')

        self.df['days_before_arrival'] = self.df['lead_time']
        self.df['total_stay'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        self.df['adr'] = pd.to_numeric(self.df['adr'], errors = 'coerce')

        self.df = self.df[
            (self.df['adr'] > 0) &
            (self.df['adr'] < 1000) &
            (self.df['days_before_arrival'] >0) &
            (self.df['days_before_arrival'] < 360) &
            (self.df['arrival_date'].notna())
        ].copy()

        print(f'Data preprocessed: {len(self.df)} valid bookings')

    """
    Estimate price elasticity(beta) from price vs occupancy relation
    Assumption: Occupancy = a * e^(-beta*price)
    Lower prices -> higher occupancy(demand)
    """
    def estimate_beta_elasticity(self):
        print("\n" + "="*80)
        print("estimating elasticity")

        price_bins = np.linspace(self.df['adr'].min(), self.df['adr'].quantile(0.95), 20)

        binned_data = pd.cut(self.df['adr'], bins = price_bins)
        occupancy_by_price = self.df.groupby(binned_data).agg({
            'adr': 'mean',
            'stays_in_weekend_nights' : 'count'
        }).reset_index(drop=True)

        occupancy_by_price.columns = ['price', 'bookings']
        occupancy_by_price = occupancy_by_price[occupancy_by_price['bookings'] > 0].copy()
        occupancy_by_price['occupancy_rate'] = (
            occupancy_by_price['bookings'] / occupancy_by_price['bookings'].max()
        )

        prices = occupancy_by_price['price'].values
        occupancy =occupancy_by_price['occupancy_rate'].values

        def demand_curve(price, a, beta):
            return a *np.exp(-beta * price)
            
        try:
            popt, pcov = curve_fit(
                demand_curve,
                prices,
                occupancy,
                p0=[1.0, 0.01],
                maxfev = 10000,
                bounds = ([0.01, 0.001], [2, 0.1])
            )
            a,beta = popt

            r_squared = 1 - (np.sum((occupancy - demand_curve(prices, a, beta))**2) /
                                 np.sum((occupancy - occupancy.mean())**2))
            print(f"\nEstimated Beta: {beta:.6f}")
            print(f"Interpretation: 1% price increase -> {beta*100:.3f}% occupancy decrease")
            print(f"R^2 (fit quality): {r_squared:.4f}")
            print(f"Coefficient a: {a:.4f}")

            self._plot_beta_elasticity(prices, occupancy, beta, a)

            return beta
        except Exception as e:
            print(f"Error bitting beta: {e}")
            print("Using default beta = 0.03")
            return 0.03
            
    """
    Estimating urgency factor(gamma) from booking velocity.
    Assumption: Demand increase as arrival date approaches
    Occupancy = a * (1+ gamma*t), where t is nomalized time to arrival
    """
    def estimate_gamma_urgency(self):
        print("\n" + "=" * 80)
        print("Estimating urgency factor")

        max_lead_time = self.df['days_before_arrival'].quantile(0.95)

        lead_time_bins = np.linspace(1, max_lead_time, 25)
        binned_data = pd.cut(self.df['days_before_arrival'], bins = lead_time_bins)
        occupancy_by_time = self.df.groupby(binned_data).agg({
            'days_before_arrival': 'mean',
            'stays_in_weekend_nights': 'count'
        }).reset_index(drop=True)

        occupancy_by_time.columns = ['days_before', 'bookings']
        occupancy_by_time = occupancy_by_time[occupancy_by_time['bookings'] > 10].copy()
        occupancy_by_time['occupancy_rate'] = (
            occupancy_by_time['bookings'] / occupancy_by_time['bookings'].max()
        )

        occupancy_by_time['t_normalized'] = (
            (max_lead_time - occupancy_by_time['days_before']) / max_lead_time
        )
        t_vals = occupancy_by_time['t_normalized'].values
        occupancy = occupancy_by_time['occupancy_rate'].values

        def urgency_curve(t, a, gamma):
            return a * (1 + gamma * t)
        
        try:
            popt, pcov = curve_fit(
                urgency_curve,
                t_vals,
                occupancy,
                p0=[0.5, 0.8],
                maxfev=10000,
                bounds=([0.1, 0.1], [2.0, 2.0])
            )
            a, gamma = popt

            r_squared = 1 - (np.sum((occupancy - urgency_curve(t_vals, a, gamma))**2) /
                                np.sum((occupancy - occupancy.mean())**2))
            print(f"\nEstimated Gamma: {gamma:.6f}")
            print(f" Interpretation: Last-minute demand {gamma*100:.1f}% higher than early bookings")
            print(f"R^2 (fit quality): {r_squared:.4f}")
            print(f"Coefficient a: {a:.4f}")   
            print(f" Max lead time used: {max_lead_time:.0f} days")

            self._plot_gamma_urgency(t_vals, occupancy, gamma, a, max_lead_time)

            return gamma
        except Exception as e:
            print(f"Error fitting gamma: {e}")
            print("using default gamma = 0.8")
            return 0.8

    """
    Estimate base demand(avg bookings/day)
    """
    def estimate_base_demand(self):
        print("\n" + "="*80)
        print("Estimating base demand")

        self.df['arrival_date_only'] = self.df['arrival_date'].dt.date
        daily_bookings = self.df.groupby('arrival_date_only').size()

        base_demand = daily_bookings.mean()
        peak_demand = daily_bookings.quantile(0.95)
        min_demand = daily_bookings.quantile(0.05)

        print(f"\nAverage daily bookings: {base_demand:.1f} rooms")
        print(f"Peak demand (95th %ile): {peak_demand:.0f} rooms/day")
        print(f"Min demand (5th %ile): {min_demand:.0f} rooms/day")
        print(f"Std deviation: {daily_bookings.std():.1f} rooms")

        self._plot_base_demand(daily_bookings)

        return base_demand, peak_demand
    
    """
    Estimate realistic price boundaries
    """
    def estimate_price_bounds(self):
        print("\n" + "+"*80)
        print("Estimating price bounds")

        p_min = self.df['adr'].quantile(0.05) 
        p_max = self.df['adr'].quantile(0.95)
        p_mean = self.df['adr'].mean()
        p_median = self.df['adr'].median()          

        print(f"\n Price minimum (5th %ile): ${p_min:.2f}")
        print(f"Price maximum (95th %ile): ${p_max:.2f}")
        print(f"Mean price: ${p_mean:.2f}")
        print(f"Median price: ${p_median:.2f}")
    
        self._plot_price_distribution()
    
        return p_min, p_max
    
    """
    Estimate total sellable inventort(rooms)
    """
    def estimate_inventory_capacity(self):
        print("\n" + "+"*80)
        print("Estimating inventory capacity")

        max_daily_bookings = self.df.groupby('arrival_date_only').size().max()

        q_max = int(max_daily_bookings * 1.5)

        print(f"Estimated hotel capacity {q_max} rooms")
        print(f"Max bookings observed: {int(max_daily_bookings)} in a single day") 
        print(f"Estimated overbooking factor: 1.5x")

        return q_max

    """
    Analyze pricing power by day of week
    """      
    def analyze_day_of_week_effect(self):
        print("\n" + "+"*80)
        print("Day of Week analysis")

        self.df['day_of_week'] = self.df['arrival_date'].dt.day_name()

        dow_analysis = self.df.groupby('day_of_week').agg({
            'adr' : ['mean', 'median', 'count'],
            'stays_in_weekend_nights': sum
        }).round(2)

        print("\n" + dow_analysis.to_string())

        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_prices = self.df.groupby('day_of_week')['adr'].mean().reindex(dow_order)

        base_price = dow_prices.mean()
        dow_premiums = ((dow_prices - base_price) / base_price  * 100).to_dict()
        print("\nPrice Premium by Day (vs average):")
        for day, premium in dow_premiums.items():
            print(f"    {day}: {premium:+.1f}%")
        
        self._plot_day_of_week(dow_prices)

        return dow_premiums
    
    """
    Analyze seasonal pricing patterns throughout the year
    """
    def analyze_seasonal_patterns(self):
        print("\n" + "+"*80)
        print("Seasonal Analysis")
        self.df['month'] = self.df['arrival_date'].dt.month
        self.df['month_name'] = self.df['arrival_date'].dt.strftime('%B')
        
        seasonal_analysis = self.df.groupby(['month', 'month_name']).agg({
            'adr': ['mean', 'median', 'std', 'count'],
        }).round(2)
        
        seasonal_analysis.columns = ['Mean Price', 'Median Price', 'Std Dev', 'Bookings']
        
        print("\n" + seasonal_analysis.to_string())
        
        seasonal_prices = self.df.groupby('month')['adr'].mean()
        base_price = seasonal_prices.mean()
        seasonal_factors = (seasonal_prices / base_price).to_dict()
        
        print("\nSeasonal Factors (multiplier vs. annual average):")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month_num, factor in seasonal_factors.items():
            print(f"  {months[month_num-1]}: {factor:.2f}x")
        
        self._plot_seasonal_patterns(seasonal_prices)
        
        return seasonal_factors
    
    """
    Analyze pricing by customer segment (market segment, customer type).
    """
    def analyze_customer_segments(self):

        print("\n" + "="*80)
        print("Customer segment analysis")
        
        if 'market_segment' in self.df.columns:
            segment_analysis = self.df.groupby('market_segment').agg({
                'adr': ['mean', 'median', 'count'],
                'stays_in_weekend_nights': 'mean'
            }).round(2)
            
            print("\nSegment Analysis:")
            print(segment_analysis)
            
            return segment_analysis
        else:
            print("✗ Market segment column not found")
            return None
        
    """
    Generate final optimized parameters for DynamicPricingEngine
    """
    def generate_final_parameters(self):

        print("\n" + "="*80)
        print("Final Parameter Estimation")
        print("="*80)
        
        beta = self.estimate_beta_elasticity()
        gamma = self.estimate_gamma_urgency()
        base_demand, peak_demand = self.estimate_base_demand()
        p_min, p_max = self.estimate_price_bounds()
        q_max = self.estimate_inventory_capacity()
        
        params = {
            'beta': beta,
            'gamma': gamma,
            'base_demand': base_demand,
            'q_max': q_max,
            'p_min': p_min,
            'p_max': p_max,
            'peak_demand': peak_demand,
        }
        
        return params
    
    def _plot_beta_elasticity(self, prices, occupancy, beta, a):
        """Plot price elasticity curve."""
        plt.figure(figsize=(12, 6))
        
        fitted_prices = np.linspace(prices.min(), prices.max(), 100)
        fitted_occupancy = a * np.exp(-beta * fitted_prices)
        
        plt.scatter(prices, occupancy, s=100, alpha=0.6, label='Historical Data', color='blue')
        plt.plot(fitted_prices, fitted_occupancy, 'r-', linewidth=2.5, label=f'Fitted Curve (β={beta:.6f})')
        
        plt.xlabel('Average Daily Rate ($)', fontsize=12, fontweight='bold')
        plt.ylabel('Normalized Occupancy Rate', fontsize=12, fontweight='bold')
        plt.title('Price Elasticity Estimation (beta)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('01_beta_elasticity.png', dpi=300, bbox_inches='tight')
        print("Saved: 01_beta_elasticity.png")
        plt.close()
    
    def _plot_gamma_urgency(self, t_vals, occupancy, gamma, a, max_lead_time):
        """Plot urgency effect curve."""
        plt.figure(figsize=(12, 6))
        
        fitted_t = np.linspace(t_vals.min(), t_vals.max(), 100)
        fitted_occupancy = a * (1 + gamma * fitted_t)
        
        plt.scatter(t_vals, occupancy, s=100, alpha=0.6, label='Historical Data', color='green')
        plt.plot(fitted_t, fitted_occupancy, 'r-', linewidth=2.5, label=f'Fitted Curve (γ={gamma:.6f})')
        
        plt.xlabel(f'Time to Arrival (normalized, 0={max_lead_time:.0f} days ago → 1=today)', fontsize=12, fontweight='bold')
        plt.ylabel('Normalized Occupancy Rate', fontsize=12, fontweight='bold')
        plt.title('Urgency Effect on Demand (gamma)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('02_gamma_urgency.png', dpi=300, bbox_inches='tight')
        print("Saved: 02_gamma_urgency.png")
        plt.close()
    
    def _plot_base_demand(self, daily_bookings):
        """Plot daily booking distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(daily_bookings, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(daily_bookings.mean(), color='red', linestyle='--', linewidth=2.5, label=f"Mean: {daily_bookings.mean():.1f}")
        axes[0].axvline(daily_bookings.median(), color='green', linestyle='--', linewidth=2.5, label=f"Median: {daily_bookings.median():.1f}")
        axes[0].set_xlabel('Daily Bookings', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of Daily Bookings', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        daily_bookings_sorted = sorted(daily_bookings)
        axes[1].plot(daily_bookings_sorted, color='darkblue', linewidth=2)
        axes[1].axhline(daily_bookings.mean(), color='red', linestyle='--', linewidth=2.5, label="Mean")
        axes[1].fill_between(range(len(daily_bookings_sorted)), daily_bookings_sorted, alpha=0.3, color='skyblue')
        axes[1].set_xlabel('Day (sorted)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Bookings', fontsize=11, fontweight='bold')
        axes[1].set_title('Daily Bookings Over Time (sorted)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('03_base_demand.png', dpi=300, bbox_inches='tight')
        print("Saved: 03_base_demand.png")
        plt.close()
    
    def _plot_price_distribution(self):
        """Plot price distribution."""
        plt.figure(figsize=(12, 6))
        
        plt.hist(self.df['adr'], bins=50, color='coral', edgecolor='black', alpha=0.7, label='Price Distribution')
        plt.axvline(self.df['adr'].mean(), color='blue', linestyle='--', linewidth=2.5, label=f"Mean: ${self.df['adr'].mean():.2f}")
        plt.axvline(self.df['adr'].median(), color='green', linestyle='--', linewidth=2.5, label=f"Median: ${self.df['adr'].median():.2f}")
        plt.axvline(self.df['adr'].quantile(0.05), color='red', linestyle=':', linewidth=2, label=f"p_min (5%): ${self.df['adr'].quantile(0.05):.2f}")
        plt.axvline(self.df['adr'].quantile(0.95), color='purple', linestyle=':', linewidth=2, label=f"p_max (95%): ${self.df['adr'].quantile(0.95):.2f}")
        
        plt.xlabel('Average Daily Rate ($)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Price Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('04_price_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: 04_price_distribution.png")
        plt.close()
    
    def _plot_day_of_week(self, dow_prices):
        """Plot day-of-week pricing."""
        plt.figure(figsize=(12, 6))
        
        colors = ['#FF6B6B' if day in ['Saturday', 'Sunday'] else '#4ECDC4' for day in dow_prices.index]
        bars = plt.bar(range(len(dow_prices)), dow_prices.values, color=colors, edgecolor='black', alpha=0.8)
        
        plt.xticks(range(len(dow_prices)), dow_prices.index, rotation=45, ha='right')
        plt.ylabel('Average Daily Rate ($)', fontsize=12, fontweight='bold')
        plt.title('Price by Day of Week', fontsize=14, fontweight='bold')
        plt.axhline(dow_prices.mean(), color='red', linestyle='--', linewidth=2, label='Average')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, price) in enumerate(zip(bars, dow_prices.values)):
            plt.text(i, price + 2, f'${price:.0f}', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('05_day_of_week.png', dpi=300, bbox_inches='tight')
        print("Saved: 05_day_of_week.png")
        plt.close()
    
    def _plot_seasonal_patterns(self, seasonal_prices):
        """Plot seasonal patterns."""
        plt.figure(figsize=(12, 6))
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors = ['#FF6B6B' if i in [6, 7] else '#4ECDC4' for i in range(1, 13)]
        
        bars = plt.bar(range(1, 13), seasonal_prices.values, color=colors, edgecolor='black', alpha=0.8)
        
        plt.xticks(range(1, 13), months, rotation=45, ha='right')
        plt.ylabel('Average Daily Rate ($)', fontsize=12, fontweight='bold')
        plt.title('Seasonal Price Patterns', fontsize=14, fontweight='bold')
        plt.axhline(seasonal_prices.mean(), color='red', linestyle='--', linewidth=2, label='Annual Average')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, price) in enumerate(zip(bars, seasonal_prices.values), 1):
            plt.text(i, price + 2, f'${price:.0f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('06_seasonal_patterns.png', dpi=300, bbox_inches='tight')
        print("Saved: 06_seasonal_patterns.png")
        plt.close()
    
if __name__ == "__main__":
    print("=" * 80)
    print("Hotel Dynamic Pricing Parameter Estimator")
    print("Dataset: Kaggle Hotel Booking Demand")
    print("=" * 80)

    try:
        estimator = HotelParameterEstimator('hotel_bookings.csv')
        params = estimator.generate_final_parameters()

        estimator.analyze_day_of_week_effect()
        estimator.analyze_seasonal_patterns()
        estimator.analyze_customer_segments()

        print("Saving Parameters to JSON")

        import json
        params_json = {
            'beta': float(params['beta']),
            'gamma': float(params['gamma']),
            'base_demand': float(params['base_demand']),
            'q_max': int(params['q_max']),
            'p_min': float(params['p_min']),
            'p_max': float(params['p_max']),
            'peak_demand': float(params['peak_demand']),
            'estimated_at': datetime.now().isoformat()
        }
        with open('hotel_pricing_parameters.json', 'w') as f:
            json.dump(params_json, f, indent=2)

        print("Parameters saved to: hotel_pricing_parameters.json")
        print("Visualizations saved: 01_beta_elasticity.png -> 06_season_patterns.png")
    except FileNotFoundError:
        print("Error: hotel_bookings.csv not found")
        