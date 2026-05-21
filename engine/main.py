import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hotel_config import load_hotel_parameters
from engine import DynamicPricingEngine, MarketplaceSimulator

CONFIG = {
    'simulation_days': 30,
    'customers_per_day': 50,
    'verbose': True,
    'save_results': True,
    'output_dir': 'results/'
}

"""
Utility Functions
"""

def create_output_directory():
    import os
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        print(f"Created output directory: {CONFIG['output_dir']}")

def print_header(title):
    print("\n" + "="*80)
    print(title.center(80))
    print("=" *80)

def print_parameters(params):
    print_header("Loaded Parameters")
    print(f"""
Price Elasticity & Demand:
    Beta (price elasticity):    {params.beta:.6f}   
    Gamma (urgency factor):     {params.gamma:.6f}
    Base demand:                {params.base_demand:.1f} bookings/day

Price Boundaries:
    Minimum price (p_min)       {params.p_min:.2f}
    Maximum price (p_max)       {params.p_max:.2f}

Inventory:
    Hotel capacity (q_max):     {int(params.q_max)} rooms
    """)

def get_optimal_price_for_scenario(engine, days_remaining, rooms_remaining):
    max_booking_window = 60
    normalized_time = days_remaining/max_booking_window

    optimal_price, expected_revenue = engine.optimize_price(
        t=normalized_time,
        q=rooms_remaining,
        method='L-BFGS-B'
    )
    return optimal_price, expected_revenue

def scenario_analysis(engine, params):
    print_header("Scenario Analysis")
    
    scenarios= [
        {'name':'Far in advance, full occupancy', 'days': 55, 'rooms': 150},
        {'name':'Mid-booking, high occupancy', 'days':30, 'rooms': 100},
        {'name':'Last-minute, medium occupancy','days': 7,'rooms':50},
        {'name':'Same-day, low occupancy','days': 1, 'rooms': 20},
        {'name':'Emergency, critical occupancy','days': 0.5,'rooms': 5},
    ]

    print("\n{:<40} {:<15} {:<20} {:<15}".format("Scenario", "Optimal Price", "Expected Revenue", "Occupancy"))
    print ("-"*90)

    for scenario in scenarios:
        optimal_price, expected_revenue = get_optimal_price_for_scenario(
            engine,
            scenario['days'],
            scenario['rooms']
        )

        occupancy_pct = (scenario['rooms'] / params.q_max) * 100

        print("{:<40} ${:<14.2f} ${:<19.2f} {:<14.1f}%".format(
            scenario['name'],
            optimal_price,
            expected_revenue,
            occupancy_pct
        ))

"""
Show how optimal price changed over booking window
"""
def price_trajectory_analysis(engine, params):
    print_header("Price Trajectory (60-Day Booking Window)")
    
    max_lead_time = 60
    rooms_starting = int(params.q_max * .8)
    
    trajectory_data =[]

    print("\n{:<12} {:<15} {:<15} {:<15}".format(
        "Days Out", "Rooms Left", "Optimal Price", "Change"
    ))
    print("-" * 60)

    previous_price = None

    for day in range(max_lead_time, 0 , -5):
        rooms_left = max(5, int(rooms_starting * (day / max_lead_time)))

        optimal_price, _ = get_optimal_price_for_scenario(engine, day, rooms_left)

        price_change = ""
        if previous_price:
            change = ((optimal_price - previous_price) / previous_price) * 100
            price_change = f"{change:+.1f}"

        print("{:<12} {:<15} ${:<14.2f} {:<15}".format(
            f"{day}d", rooms_left, optimal_price, price_change
        ))

        trajectory_data.append({
            'days_out': day,
            'rooms_left': rooms_left,
            'optimal_price': optimal_price
        })

        previous_price = optimal_price
    return pd.DataFrame(trajectory_data)

def full_simulation(engine, params):
    print_header("30-Day Marketplace Simulation")

    simulator = MarketplaceSimulator(
        engine,
        num_days=CONFIG['simulation_days']
    )

    results_df = simulator.run_simulation(verbose=CONFIG['verbose'])

    if CONFIG['save_results']:
        results_path = f"{CONFIG['output_dir']}simulation_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        
        print("\nGenerating visualizations")
        
        fig1_path = f"{CONFIG['output_dir']}simulation_4plots.png"
        fig1 = simulator.plot_results()
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig1_path}")
        
        fig2_path = f"{CONFIG['output_dir']}revenue_surface.png"
        fig2 = simulator.plot_revenue_surface()
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig2_path}")
    
    return results_df, simulator

def compare_strategies(engine, params):
    print_header("Strategy Comparison: Dynamic vs Static Pricing")

    max_lead_time = 60
    rooms_available = int(params.q_max * 0.9)
    static_price= (params.p_min + params.p_max) / 2

    dynamic_revenue = 0
    static_revenue = 0

    print("\n{:<15} {:<18} {:<18} {:<15}".format(
        "Lead Time", "Dynamic Price", "Static Price", "Revenue Diff"
    ))
    print("-" * 70)

    for day in range(max_lead_time, 0 , -5):
        rooms_left = max(5, int(rooms_available * (day / max_lead_time)))

        optimal_price, dynamic_rev = get_optimal_price_for_scenario(engine, day, rooms_left)
        static_rev = engine.revenue_function(static_price, day/max_lead_time, rooms_left)

        dynamic_revenue += dynamic_rev
        static_revenue += static_rev

        diff = ((dynamic_rev - static_rev) / (static_rev +1)) * 100

        print("{:<15} ${:<17.2f} ${:<17.2f} {:<14.1f}%".format(
            f"{day}d", optimal_price, static_price, diff
        ))

    uplift = ((dynamic_revenue - static_revenue) / static_revenue) * 100

    print("\n" + "-" * 70)
    print("Total Revenue Comparison:")
    print(f"  Dynamic Pricing: ${dynamic_revenue:.2f}")
    print(f"  Static Pricing:  ${static_revenue:.2f}")
    print(f"  Revenue Uplift:  {uplift:+.1f}% (${dynamic_revenue - static_revenue:+.2f})")

def interactive_pricing_tool(engine, params):
    print_header("Interactive Pricing Tool")

    while True:
        try:
            print("\nEnter scenario (or 'exit' to return to menu):")
            user_input = input("  ").strip().lower()

            if user_input == 'exit':
                break

            parts = user_input.split()
            if len(parts) != 2:
                print("Format: <days_remaining> <rooms_remaining>")
                continue

            days = float(parts[0])
            rooms = int(parts[1])

            if days < 0 or rooms <0:
                print("Please enter positive numbers")
                continue

            if rooms > params.q_max:
                print(f"Maximum rooms is {int(params.q_max)}")
                continue

            optimal_price, expected_revenue = get_optimal_price_for_scenario(engine, days, rooms)

            print(f"\n Result:")
            print(f"Days until arrival: {days}")
            print(f"Rooms Remaining: {rooms}")
            print(f"    Recommended price: ${optimal_price:.2f}")
            print(f"    Expected Revenue: ${expected_revenue:.2f}")
        except ValueError:
            print("Invalid Input. Try again")
        except Exception as e:
            print(f" Error: {e}")

def print_menu():
    print_header("Hotel Dynamic Pricing Engine - Main Menu")
    print("""
1. View loaded parameters
2. Scenario analysis (predefined scenarios)
3. Price trajectory (60-day booking window)
4. Full 30-day simulation 
5. Strategy comparison (dynamic vs static pricing)
6. Interactive pricing tool
7. Exit
    """)

def main():
    print_header("Hotel Dynamic Pricing Engine")
    print("loading system...")

    try:
        params = load_hotel_parameters()
        print("Parameters loaded successfully")

        engine = DynamicPricingEngine(params)
        print("Pricing engine initialized")

        create_output_directory()
        print_parameters(params)

        while True:
            print_menu()
            choice = input("Enter Choice (1-7):").strip()

            if choice == '1':
                print_parameters(params)
            elif choice == '2':
                scenario_analysis(engine, params)
            elif choice == '3':
                trajectory_df = price_trajectory_analysis(engine, params)
                if CONFIG['save_results']:
                    trajectory_df.to_csv(
                        f"{CONFIG['output_dir']}price_trajectory.csv",
                        index=False
                    )
                    print("Trajectory saved to results/price_trajectory.csv")

            elif choice == '4':
                results_df, simulator = full_simulation(engine, params)
            elif choice == '5':
                compare_strategies(engine, params)
            elif choice == '6':
                interactive_pricing_tool(engine, params)
            elif choice == '7':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Select 1-7.")

    except FileNotFoundError as e:
        print(f"\n Error: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()