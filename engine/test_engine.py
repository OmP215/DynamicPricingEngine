import pytest
import numpy as np
import pandas as pd
from engine import (
    PricingParams,
    DynamicPricingEngine,
    CustomerAgent,
    MarketplaceSimulator
)

"""
Tests for PricingParams
"""

class TestPricingParams:
    """Test parameter initialization and validation"""

    def test_default_parameters(self):
        params = PricingParams()

        assert params.beta == 0.05
        assert params.gamma == 0.8
        assert params.q_max == 100.0
        assert params.p_min == 10.0
        assert params.p_max == 100.0
        assert params.base_demand == 100.0

    def test_custom_parameters(self):
        """Test that custom parameters can be set"""
        params = PricingParams(
            beta=0.03,
            gamma=0.9,
            q_max=200.0,
            p_min=50.0,
            p_max=300.0,
            base_demand=75.0
        )


        assert params.beta == 0.03
        assert params.gamma == 0.9
        assert params.q_max == 200.0
        assert params.p_min == 50.0
        assert params.p_max == 300.0
        assert params.base_demand == 75.0

    def test_price_min_max_relationship(self):
        """Test that p_min < p_max"""
        params = PricingParams()
        assert params.p_min < params.p_max

    def test_parameters_are_positive(self):
        """Test that all parameters are > 0"""
        params = PricingParams()

        assert params.beta > 0
        assert params.gamma > 0
        assert params.q_max > 0
        assert params.p_min > 0
        assert params.p_max > 0
        assert params.base_demand > 0

"""
Tests for DemandFunction
"""
class TestDemandFunction:
    """Test demand calculations"""

    @pytest.fixture
    def engine(self):
        """Create engine instance for tests"""
        params = PricingParams(beta=0.05,gamma=0.8, base_demand=100.0)
        return DynamicPricingEngine(params)
    
    def test_demand_positive(self, engine):
        """Demand should always be > 0"""
        demand = engine.demand_function(p=20,t=0.5,q=100)
        assert demand > 0

    def test_demand_decreases_with_price(self,engine):
        """Higher price should result in lower beta"""
        demand_low_price = engine.demand_function(p=10,t=0.5,q=100)
        demand_high_price = engine.demand_function(p=50, t=0.5 ,q=100)
        assert demand_low_price > demand_high_price

    def test_demand_increases_with_urgency(self, engine):
        """Demand should increase as time to arrival approaches"""
        demand_early = engine.demand_function(p=20, t= 0.1, q=100)
        demand_late = engine.demand_function(p=20,t=0.9,q=100)
        assert demand_late > demand_early

    def test_demand_increases_with_inventory(self, engine):
        """More inv should not decrease demand(up to q_max)"""
        demand_low_inv = engine.demand_function(p=20,t=0.5,q=10)
        demand_high_inv = engine.demand_function(p=20, t=0.5, q=100)
        assert demand_high_inv >= demand_low_inv

    def test_demand_zero_inventory(self, engine):
        """Demand w/ 0 inventory should be 0"""
        demand = engine.demand_function(p=20, t=0.5, q=0)
        assert demand ==0

    def test_demand_bounded_by_inventory(self, engine):
        """Demand should be capped by inv constraints"""
        demand = engine.demand_function(p=20, t=0.5, q=1000)

        #Max theoretical demand
        #no price sensitivity, full urgency, full inventory
        max_demand = engine.params.base_demand * np.exp(0) * (1 + engine.params.gamma * 1.0) * 1.0
        assert max_demand >= demand

"""
Tests for RevenueFunction
"""
class TestRevenueFunction:
    """Test revenue calculations"""

    @pytest.fixture
    def engine(self):
        params = PricingParams(beta=0.05, gamma=0.8, base_demand=100.0)
        return DynamicPricingEngine(params)
    
    def test_revenue_formula(self, engine):
        """Revenue should equal price * demand"""
        price = 20
        t=0.5
        q=50

        demand = engine.demand_function(price, t, q)
        revenue = engine.revenue_function(price, t, q)

        assert revenue == pytest.approx(price*demand)

    def test_revenue_positive(self, engine):
        """Reveenue should be pos for all valid inputs"""
        revenue = engine.revenue_function(p=20,t=0.5,q=50)
        assert revenue >0

    def test_revenue_zero_inventory(self, engine):
        """Revenue w/ 0 inv should be 0"""
        revenue = engine.revenue_function(p=20,t=0.5,q=0)
        assert revenue == 0

    def test_revenue_increases_with_beta(self, engine):
        """Test that optimal price exists between pmin and pmax"""
        rev_at_min = engine.revenue_function(p=engine.params.p_min, t=0.5, q=100)
        rev_at_max = engine.revenue_function(p=engine.params.p_max, t=0.5, q=100)
        rev_at_mid = engine.revenue_function(
            p=(engine.params.p_min + engine.params.p_max) / 2,
            t= 0.5,
            q=100
        )
        assert rev_at_mid > min(rev_at_min, rev_at_max)

"""
Tests for Gradients
"""
class TestGradients:
    """Test gradient computations"""
    @pytest.fixture
    def engine(self):
        params = PricingParams(beta=0.05, gamma=0.8, base_demand=100.0)
        return DynamicPricingEngine(params)
    
    def test_gradients_return_tuple(self, engine):
        """Gradients should return (dR/dp, dR/dt, dR/dq)"""
        result = engine.compute_gradients(p=20,t=0.5,q=50)
        assert isinstance(result, tuple)
        assert len(result) ==3

    def test_price_gradient_sign(self,engine):
        """Price gradients should be pos at optimal price point"""
        p= 15
        dR_dp, _, _ = engine.compute_gradients(p=p, t=0.5, q=100)
        assert dR_dp > 0

    def test_time_gradient_positive(self, engine):
        """Time gradient should be pos"""
        dR_dp, dR_dt, dR_dq = engine.compute_gradients(p=20, t=0.5, q=100)
        assert dR_dt > 0

    def test_inventory_gradient_positive(self, engine):
        """Inv gradient should be pos"""
        dR_dp, dR_dt, dR_dq = engine.compute_gradients(p=20, t=0.5, q=100)
        assert dR_dq > 0
    
"""
Tests for Hessian
"""
class TestHessian:
    """Test hessian matrix computations"""
    @pytest.fixture
    def engine(self):
        params = PricingParams(beta=0.05, gamma=0.8, base_demand=100.0)
        return DynamicPricingEngine(params)

    def test_hessian_shape(self, engine):
        """Should be 3x3 matrix"""
        hessian = engine.compute_hessian(p=20,t=0.5,q=50)
        assert hessian.shape == (3,3)

    def test_hessian_symmetry(self, engine):
        """Mixed partials are equal(symmetric)"""
        hessian = engine.compute_hessian(p=20,t=0.5,q=50)

        assert hessian[0,1] == pytest.approx(hessian[1,0])
        assert hessian[0,2] == pytest.approx(hessian[2,0])  
        assert hessian[1,2] == pytest.approx(hessian[2,1])

    def test_hessian_price_concavity(self, engine):
        """Second derivativr w.r.t price should be neg(concave)"""
        hessian = engine.compute_hessian(p=20,t=0.5,q=50)
        #H[0,0] is d^2R/dp^2
        assert hessian[0,0] < 0

"""
Tests for OptimizePrice
"""
class TestOptimizePrice:
    """Test price optimization"""

    @pytest.fixture
    def engine(self):
        params = PricingParams(
            beta=0.05, 
            gamma=0.8, 
            base_demand=100.0,
            p_min=10.0,
            p_max=100.0            
        )
        return DynamicPricingEngine(params)
    
    def test_optimal_price_within_bounds(self, engine):
        """Optimal price should be within pmin and pmax"""
        optimal_price, _ = engine.optimize_price(t=0.5, q=50)
        assert engine.params.p_min <= optimal_price <= engine.params.p_max

    def test_optimal_price_pos(self, engine):
        """OPtimal price should be pos"""
        optimal_price, _ = engine.optimize_price(t=0.5, q=50)
        assert optimal_price > 0

    def test_expected_revenue_pos(self, engine):
        """Expected revenue should be pos for pos inv"""
        _, expected_revenue = engine.optimize_price(t=0.5,q=50)

        assert expected_revenue > 0

    def test_zero_inv_returns_min_price(self, engine):
        """w/0 inv, should return pmin and 0 revenue"""
        optimal_price, expected_revenue = engine.optimize_price(t=0.5, q=0)
        assert optimal_price == engine.params.p_min
        assert expected_revenue == 0.0

    def test_optimization_history_recorded(self, engine):
        """each optimization should be recorded in history"""
        initial_d = len(engine.optimization_history)
        engine.optimize_price(t=0.5,q=50)
        assert len(engine.optimization_history) == initial_d +1

    def test_optimal_price_near_theoretical(self, engine):
        """Optimal price should be near 1/beta"""
        optimal_price, _ = engine.optimize_price(t=0.5,q=100)
        theoretical_optimal = 1.0 / engine.params.beta

        #should be 10% of theoretical
        assert abs(optimal_price - theoretical_optimal) / theoretical_optimal <0.1

"""
Tests for CustomerAgent
"""
class TestCustomerAgent:
    """Test customer behavior"""

    def test_purchase_decision_deterministic_bounds(self):
        """Test purchase probability bounds (0 to 1)"""
        customer = CustomerAgent(
            elasticity=0.15,
            value_multiplier=1.0,
            willingness_to_pay=50.0
        )

        decisions = [customer.purchase_decision(50.0) for _ in range(100)]

        assert all(isinstance(d, (bool, np.bool_)) for d in decisions)

    def test_higher_price_fewer_purchases(self):
        """Higher price should result in fewer purchases"""
        customer = CustomerAgent(
            elasticity=0.5,
            value_multiplier=1.0,
            willingness_to_pay=50.0
        )

        purchases_low_price = sum(
            customer.purchase_decision(current_price=40.0)
            for _ in range(100)
        )

        purchases_high_price = sum(
            customer.purchase_decision(current_price=80.0)
            for _ in range(100)
        )
        assert purchases_low_price > purchases_high_price

    def test_purchase_at_wtp(self):
        """Purchase probability at wtp should be ~50%"""
        customer = CustomerAgent(
            elasticity=1.0,
            value_multiplier=1.0,
            willingness_to_pay=50.0
        )

        purchases = sum(
            customer.purchase_decision(current_price=50.0)
            for _ in range(1000)
        )

        purchase_rate = purchases / 1000
        assert 0.4 < purchase_rate < 0.6

"""
Tests for MarketplaceSimulator
"""
class TestMarketplaceSimulator:
    """Test simulation functionality"""
    @pytest.fixture
    def simulator(self):
        params = PricingParams(
            beta=0.05,
            gamma=0.8,
            q_max=100.0,
            p_min=10.0,
            p_max=100.0,
            base_demand=100.0
        )
        engine = DynamicPricingEngine(params)
        return MarketplaceSimulator(engine, num_days=5)
    
    def test_simulation_runs_without_errors(self, simulator):
        """Sim should run w/o crashing"""
        results_df = simulator.run_simulation(verbose=False)
        assert results_df is not None

    def test_simulation_returns_df(self, simulator):
        """Sim should return a pd DF"""
        results_df = simulator.run_simulation(verbose = False)
        assert isinstance(results_df, pd.DataFrame)

    def test_simulation_has_correct_rows(self,simulator):
        """One row/day"""
        results_df = simulator.run_simulation(verbose = False)
        assert len(results_df) == simulator.num_days       
    
    def test_simulation_has_required_columns(self, simulator):
        """Should have all columns"""
        results_df = simulator.run_simulation(verbose = False)

        required_columns = [
            'day', 'time_remaining', 'inventory_start', 'inventory_end',
            'optimal_price', 'expected_demand', 'actual_sales',
            'daily_revenue', 'cumulative_revenue', 'utilization'
        ]

        for col in required_columns:
            assert col in results_df.columns     
    
    def test_inventory_decreases_monotonically(self, simulator):
        """Inv should never increase"""
        results_df = simulator.run_simulation(verbose = False)

        inventories = results_df['inventory_end'].values
        for i in range(1, len(inventories)):
            assert inventories[i] <= inventories[i-1]

    def test_cumulative_revenue_increases(self, simulator):
        """Cumulative revenue should never decrease"""
        results_df = simulator.run_simulation(verbose = False)
        cumulative_revenues = results_df['cumulative_revenue'].values
        for i in range(1, len(cumulative_revenues)):
            assert cumulative_revenues[i] >= cumulative_revenues[i-1]

    def test_prices_within_bounds(self, simulator):
        """all prices should be within pmin and pmax"""
        results_df = simulator.run_simulation(verbose = False)
        prices = results_df['optimal_price'].values
        p_min = simulator.engine.params.p_min
        p_max = simulator.engine.params.p_max

        assert np.all(prices >= p_min)
        assert np.all(prices <= p_max)

    def test_daily_revenue_pos(self,simulator):
        """Daily revenue should be non neg"""
        results_df = simulator.run_simulation(verbose = False)
        assert np.all(results_df['daily_revenue'].values >= 0)

"""
Integration Tests
"""
class TestIntegration:
    """Test full system integration"""

    def test_full_workflow(self):
        params = PricingParams(
            beta=0.05,
            gamma=0.8,
            q_max=100.0,
            p_min=10.0,
            p_max=100.0,
            base_demand=100.0     
        )

        engine = DynamicPricingEngine(params)
        optimal_price, revenue = engine.optimize_price(t=0.5,q=50)
        assert optimal_price is not None
        assert revenue >0

        simulator = MarketplaceSimulator(engine, num_days=10)
        results_df = simulator.run_simulation(verbose = False)
        assert len(results_df) == 10
        assert results_df['cumulative_revenue'].iloc[-1] > 0

    def test_different_parameter_sets(self):
        param_sets = [
            {'beta':0.01, 'gamma':0.5},
            {'beta':0.10, 'gamma':1.5},
            {'beta':0.05, 'gamma':0.8},
        ]

        for param_dict in param_sets:
            params = PricingParams(**param_dict)
            engine = DynamicPricingEngine(params)

            optimal_price, revenue = engine.optimize_price(t=0.5,q=50)
            assert optimal_price is not None
            assert revenue >= 0

"""
Edge Case Tests
"""
class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    @pytest.fixture
    def engine(self):
        params = PricingParams(beta=0.05, gamma=0.8, base_demand=100.0)
        return DynamicPricingEngine(params)

    def test_small_inventory(self, engine):
        optimal_price, revenue = engine.optimize_price(t=0.5,q=1)
        assert optimal_price > 0
        assert revenue >= 0

    def test_large_inventory(self, engine):
        optimal_price, revenue = engine.optimize_price(t=0.5,q=10000)
        assert optimal_price > 0
        assert revenue >= 0

    def test_time_boundaries(self, engine):
        # At t=0 (far future)
        price_early, revenue_early = engine.optimize_price(t=0,q=100)

        # At t=1 (now)
        price_now, revenue_now = engine.optimize_price(t=1,q=100)
        assert price_early > 0
        assert price_now > 0

        assert price_now >= price_early - 1e6
        assert revenue_now > revenue_early

    def test_extreme_parameters(self):
        extreme_params = PricingParams(
            beta=0.001,
            gamma=2.0,
            q_max=1000.0,
            p_min=1.0,
            p_max=10000.0,
            base_demand=1000.0
        )

        engine = DynamicPricingEngine(extreme_params)
        optimal_price, revenue = engine.optimize_price(t=0.5,q=500)

        assert optimal_price > 0
        assert revenue >= 0

"""
Run Tests
"""
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])