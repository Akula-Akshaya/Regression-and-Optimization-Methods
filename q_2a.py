import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading earth quake data from the csv file
df = pd.read_csv("earthquake_alert_balanced_dataset.csv")

#to prepare data
X = df[['magnitude', 'depth', 'cdi', 'mmi']].values
y = df['alert'].map({'green': 0, 'yellow': 1, 'orange': 2, 'red': 3}).values
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
X = np.c_[X, np.ones(len(X))]  #adding bias term

# Converting to regression tar
y = y / 3.0

print("="*70)
print("EARTHQUAKE ALERT PREDICTION OPTIMIZATION")
print("="*70)
print(f"Dataset: {len(X)} samples, {X.shape[1]-1} features + bias")
print()

#unconstrained opt
print("\n" + "="*70)
print("UNCONSTRAINED CASE")
print("="*70)
print("Objective: min f(w) = (1/n) Σ(y_i - w^T x_i)^2 + λ||w||^2")
print("Method: Gradient Descent")
print()

def objective(w, X, y, lam=0.01):
    pred = X @ w
    mse = np.mean((y - pred)**2)
    reg = lam * np.sum(w[:-1]**2)
    return mse + reg

def gradient(w, X, y, lam=0.01):
    pred = X @ w
    grad = -2 * X.T @ (y - pred) / len(X)
    grad[:-1] += 2 * lam * w[:-1]
    return grad

#gradient descent
w = np.zeros(X.shape[1])
lr = 0.1
tol = 1e-6
max_iter = 100

print("Update formula: w_(k+1) = w_k - α*∇f(w_k)")
print(f"Step size: α = {lr}")
print(f"Stopping criterion: ||∇f|| < {tol}")
print()

#storing history to plot
gd_obj_history = []
gd_grad_norm_history = []
gd_iterations = []

for i in range(max_iter):
    grad = gradient(w, X, y)
    obj = objective(w, X, y)
    grad_norm = np.linalg.norm(grad)
    
    gd_obj_history.append(obj)
    gd_grad_norm_history.append(grad_norm)
    gd_iterations.append(i)
    
    if i % 20 == 0:
        print(f"Iter {i:3d}: Objective = {obj:.6f}, ||∇f|| = {grad_norm:.8f}")
    
    if grad_norm < tol:
        print(f"Converged at iteration {i}")
        break
    
    w = w - lr * grad

print(f"\nFinal objective: {objective(w, X, y):.6f}")
print(f"Optimal weights: {w}")
print(f"Feature importance: magnitude={w[0]:.4f}, depth={w[1]:.4f}, cdi={w[2]:.4f}, mmi={w[3]:.4f}")
print()
#newton's mthd
print("\n" + "-"*70)
print("Method: Newton's Method")
print()

def hessian(X, lam=0.01):
    H = 2 * X.T @ X / len(X)
    H[:-1, :-1] += 2 * lam * np.eye(X.shape[1]-1)
    return H

w_newton = np.zeros(X.shape[1])
max_iter_newton = 50

print("Update formula: w_(k+1) = w_k - H^(-1)*∇f(w_k)")
print("Hessian: H = 2*X^T*X/n + 2λI")
print()

#storing his to plot
newton_obj_history = []
newton_grad_norm_history = []
newton_iterations = []

for i in range(max_iter_newton):
    grad = gradient(w_newton, X, y)
    H = hessian(X)
    obj = objective(w_newton, X, y)
    grad_norm = np.linalg.norm(grad)
    
    newton_obj_history.append(obj)
    newton_grad_norm_history.append(grad_norm)
    newton_iterations.append(i)
    
    if i % 10 == 0:
        print(f"Iter {i:3d}: Objective = {obj:.6f}, ||∇f|| = {grad_norm:.8f}")
    
    if grad_norm < tol:
        print(f"Converged at iteration {i}")
        break
    
    w_newton = w_newton - np.linalg.solve(H, grad)

print(f"\nFinal objective: {objective(w_newton, X, y):.6f}")
print(f"Optimal weights: {w_newton}")
print()

#constrained opt
print("\n" + "="*70)
print("CONSTRAINED CASE")
print("="*70)
print("Objective: min f(w) = (1/n) Σ(y_i - w^T x_i)^2")
print("Constraints: g(w) = ||w||^2 - 1 ≤ 0  (inequality)")
print("             h(w) = Σw_i = 0        (equality)")
print()

#using penalty mthd
print("\n" + "-"*70)
print("METHOD 1: PENALTY METHOD")
print("="*70)
print("Penalty function: P(w,ρ) = f(w) + ρ*max(0, g(w))^2 + ρ*h(w)^2")
print("Update rule: ρ_(k+1) = 1.5*ρ_k")
print()

def penalty_objective(w, X, y, rho):
    obj = objective(w, X, y, lam=0)
    w_features = w[:-1]
    g = np.sum(w_features**2) - 1  # ||w||^2 - 1 <= 0
    h = np.sum(w_features)          # sum(w) = 0
    penalty = rho * (max(0, g)**2 + h**2)
    return obj + penalty

w_penalty = np.random.randn(X.shape[1]) * 0.01
rho = 0.1
lr_penalty = 0.01

print("Starting penalty method...")
penalty_obj_history = []
penalty_constraint_history = []
penalty_rho_history = []
penalty_outer_iters = []

for outer in range(15):
    for inner in range(20):
        pred = X @ w_penalty
        grad_obj = -2 * X.T @ (y - pred) / len(X)
        
        w_features = w_penalty[:-1]
        g = np.sum(w_features**2) - 1
        h = np.sum(w_features)
        
        grad_penalty = np.zeros_like(w_penalty)
        if g > 0:
            grad_penalty[:-1] += 2 * rho * g * 2 * w_features
        grad_penalty[:-1] += 2 * rho * h
        
        grad_total = grad_obj + grad_penalty
        grad_norm = np.linalg.norm(grad_total)
        
        step_size = lr_penalty / max(1, grad_norm / 10)
        w_penalty = w_penalty - step_size * grad_total
        w_penalty = np.clip(w_penalty, -10, 10)
    
    obj = penalty_objective(w_penalty, X, y, rho)
    w_features = w_penalty[:-1]
    g = np.sum(w_features**2) - 1
    h = np.sum(w_features)
    viol = np.sqrt(max(0, g)**2 + h**2)
    
    penalty_obj_history.append(obj)
    penalty_constraint_history.append(viol)
    penalty_rho_history.append(rho)
    penalty_outer_iters.append(outer)
    
    if outer % 3 == 0:
        print(f"Outer iter {outer}: ρ={rho:.4f}, Obj={obj:.6f}, Constraint viol={viol:.6f}")
    
    rho *= 1.5

print(f"\nPenalty Method Results:")
print(f"Final weights: {w_penalty}")
print(f"||w||^2 = {np.sum(w_penalty[:-1]**2):.6f} (should be ≤ 1)")
print(f"Σw_i = {np.sum(w_penalty[:-1]):.6f} (should be ≈ 0)")
print()


# 4
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Earthquake Alert Prediction - Optimization Analysis', fontsize=16, fontweight='bold')

#1-gradient descent convergence
ax1 = axes[0, 0]
ax1.plot(gd_iterations, gd_obj_history, 'b-', linewidth=2, label='Objective')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Objective Value')
ax1.set_title('Gradient Descent: Objective Convergence')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax1b = ax1.twinx()
ax1b.plot(gd_iterations, gd_grad_norm_history, 'r--', linewidth=1.5, label='Gradient Norm')
ax1b.set_ylabel('Gradient Norm (log scale)', color='red')
ax1b.set_yscale('log')
ax1b.tick_params(axis='y', labelcolor='red')

#2-newton's mthd convergence
ax2 = axes[0, 1]
ax2.plot(newton_iterations, newton_obj_history, 'g-', linewidth=2, marker='o', markersize=4)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Objective Value')
ax2.set_title("Newton's Method: Objective Convergence")
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.9, f'Converged in {len(newton_iterations)} iterations', 
         transform=ax2.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))


#3-penalty mthd convergence
ax3 = axes[1, 0]
ax3.plot(penalty_outer_iters, penalty_obj_history, 'm-', linewidth=2, label='Penalized Objective')
ax3.set_xlabel('Outer Iteration')
ax3.set_ylabel('Objective Value')
ax3.set_title('Penalty Method: Objective vs Constraints')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')

ax3b = ax3.twinx()
ax3b.plot(penalty_outer_iters, penalty_constraint_history, 'c--', linewidth=1.5, label='Constraint Violation')
ax3b.set_ylabel('Constraint Violation', color='cyan')
ax3b.tick_params(axis='y', labelcolor='cyan')
ax3b.legend(loc='upper right')

#4-feature imp comparision
ax4 = axes[1, 1]
features = ['Magnitude', 'Depth', 'CDI', 'MMI', 'Bias']
x_pos = np.arange(len(features))
width = 0.25

# excluding bias and getting weights
gd_weights = w[:4]
newton_weights = w_newton[:4]
penalty_weights = w_penalty[:4]

# Pading with zeroes for bias term
gd_all = list(gd_weights) + [w[4]]
newton_all = list(newton_weights) + [w_newton[4]]
penalty_all = list(penalty_weights) + [w_penalty[4]]

ax4.bar(x_pos - width, gd_all, width, label='Gradient Descent', alpha=0.8)
ax4.bar(x_pos, newton_all, width, label="Newton's Method", alpha=0.8)
ax4.bar(x_pos + width, penalty_all, width, label='Penalty Method', alpha=0.8)

ax4.set_xlabel('Features')
ax4.set_ylabel('Weight Value')
ax4.set_title('Feature Importance Comparison Across Methods')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(features, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('optimization_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'optimization_analysis.png'")

plt.show()

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print("1. Gradient Descent: Slow but steady convergence")
print("2. Newton's Method: Fast convergence (1-2 iterations)")
print("3. Penalty Method: Trade-off between objective and constraints")
print("4. Feature Importance: MMI is most important predictor")
print("="*70)