import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

## Simulate Demand

def simulate_demand(T = 100, noise_level = 5):
	t = np.arange(T)
	demand = 50 + 20 * np.sin(2*np.pi * t /24) ## a daily cycle
	noise = np.random.normal(0, noise_level, T)
	return demand + noise 


## Define Generation Capacity

def generation_capacity(T = 100, capacity = 60):
	return np.full(T,capacity)

## Compute Metrics
def compute_metrics(demand, generation):
	unmet = np.maximum(demand - generation, 0)
	excess = np.maximum(generation - demand, 0)

	return pd.DataFrame({
		"demand": demand, 
		"generation": generation,
		"unmet_demand": unmet,
		"excess_energy": excess
		})


## Run simulation

T = 100
demand = simulate_demand(T)
generation = generation_capacity(T)

df = compute_metrics(demand, generation)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10,5))
plt.plot(df["demand"], label="Demand")
plt.plot(df["generation"], label="Generation")
plt.legend()
plt.title("Energy Demand vs Generation")
plt.savefig("Images/demand_vs_generation.png", dpi = 300, bbox_inches="tight")






import cvxpy as cp

def optimize_generation(demand, capacity, cost=1, penalty=10):
    T = len(demand)
    
    G = cp.Variable(T)
    
    objective = cp.Minimize(
        cost * cp.sum(G) +
        penalty * cp.sum(cp.pos(demand - G))
    )
    
    constraints = [
        G >= 0,
        G <= capacity
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return G.value


capacity = 60  # max generation

G_opt = optimize_generation(demand, capacity)

df["optimized_generation"] = G_opt
df["optimized_unmet"] = np.maximum(demand - G_opt, 0)

## optimized plot 

plt.figure(figsize=(10,5))
plt.plot(df["demand"], label="Demand")
plt.plot(df["optimized_generation"], label="Optimized Generation")
plt.legend()
plt.title("Energy Demand vs Optimized Generation")
plt.savefig("Images/demand_vs_optimized_generation.png", dpi = 300, bbox_inches = "tight")


## optimized vs fixed generation
plt.figure(figsize=(10,5))
plt.plot(df["generation"], label="Generation")
plt.plot(df["optimized_generation"], label="Optimized Generation")
plt.legend()
plt.title("Fixed Generation vs Optimized Generation")
plt.savefig("Images/fixed_vs_optimized_generation.png", dpi = 300, bbox_inches = "tight")


def total_cost(demand, generation, cost=1.0, penalty=10.0):
    unmet = np.maximum(demand - generation, 0)
    return np.sum(cost * generation + penalty * unmet)

cost_fixed = total_cost(df["demand"], df["generation"])
cost_opt = total_cost(df["demand"], df["optimized_generation"])

savings = cost_fixed - cost_opt
percent_savings = 100 * savings / cost_fixed

print(f"Fixed cost: {cost_fixed:.2f}")
print(f"Optimized cost: {cost_opt:.2f}")
print(f"Savings: {savings:.2f} ({percent_savings:.2f}%)")