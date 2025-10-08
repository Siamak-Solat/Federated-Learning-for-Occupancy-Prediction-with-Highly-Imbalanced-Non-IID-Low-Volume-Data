# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
server_coordinator_baseline.py - ABLATED BASELINE FEDAVG VERSION

REMOVED TECHNIQUES:
1. FedProx proximal correction → Simple weight averaging (FedAvg)
2. Dynamic μ computation → No proximal term
3. Inter-client dispersion calculation → Not needed for simple averaging

KEPT:
- Federated learning with simple averaging (FedAvg)
- Same number of rounds (5)
- Global metrics tracking

WHY: To measure the impact of FedProx vs simple FedAvg
"""

import os, time, yaml, json
import numpy as np
import tensorflow as tf
import absl.logging as absl_logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

CFG = yaml.safe_load(open("all_zones_baseline.yaml"))

ROUNDS = CFG["common"]["federated"]["rounds"]  # Still 5 rounds
ZONES = list(CFG["zones"].keys())
SLEEP = 10

os.makedirs("global_weights", exist_ok=True)
if not os.path.exists("global_metrics_baseline.csv"):
    with open("global_metrics_baseline.csv", "w") as f:
        # REMOVED: mu column since we don't use it
        f.write("round,train_loss,val_loss\n")

def all_exist(paths):
    return all(os.path.exists(p) for p in paths)

def average_weight_sets(wsets):
    """ABLATED BASELINE: Basic averaging without any weighting"""
    return [np.mean([w[i] for w in wsets], axis=0)
            for i in range(len(wsets[0]))]

# REMOVED: compute_dispersion function - not needed for simple FedAvg
# REMOVED: dynamic_mu function - not needed for simple FedAvg

def compute_global_losses(metric_paths):
    """Weighted average of losses based on sample counts"""
    tot_tr = tot_va = n_tr = n_va = 0
    for p in metric_paths:
        with open(p) as f:
            m = json.load(f)
        tot_tr += m["train_loss"] * m["n_train"]
        tot_va += m["val_loss"] * m["n_val"]
        n_tr += m["n_train"]
        n_va += m["n_val"]
    return (tot_tr/n_tr if n_tr else float("nan"),
            tot_va/n_va if n_va else float("nan"))

def log_global_losses(rnd, g_tr, g_va):
    """ABLATED BASELINE: No μ parameter to log"""
    print(f"Global losses | round {rnd}: "
          f"train={g_tr:.4f}  val={g_va:.4f}")
    with open("global_metrics_baseline.csv", "a") as f:
        f.write(f"{rnd},{g_tr},{g_va}\n")

# ───────── Round 0 ─────────
print("=" * 60)
print("BASELINE SERVER - Simple FedAvg (no FedProx)")
print("=" * 60)
print("\nServer waiting for all clients to finish round 0 ...")

r0_models = [f"{z}_round0_baseline.keras" for z in ZONES]
r0_metrics = [f"{z}_metrics_round0_baseline.json" for z in ZONES]

while not all_exist(r0_models + r0_metrics):
    time.sleep(SLEEP)

print("All round-0 files detected. Aggregating with simple averaging...")

# Load all models and average their weights
w_sets = [tf.keras.models.load_model(f, compile=False).get_weights()
          for f in r0_models]

# ABLATED BASELINE: Basic averaging without FedProx correction
avg = average_weight_sets(w_sets)

# Use first model as template for architecture
tmpl = tf.keras.models.load_model(r0_models[0], compile=False)
tmpl.set_weights(avg)
tmpl.save("global_weights/global_round0_baseline.keras")

# Compute and log global losses
g_tr, g_va = compute_global_losses(r0_metrics)
log_global_losses(0, g_tr, g_va)

# ───────── Federated rounds 1..R ─────────
for r in range(1, ROUNDS + 1):
    print(f"\nRound {r}/{ROUNDS} – waiting for uploads ...")
    
    r_models = [f"{z}_round{r}_baseline.keras" for z in ZONES]
    r_metrics = [f"{z}_metrics_round{r}_baseline.json" for z in ZONES]
    
    while not all_exist(r_models + r_metrics):
        time.sleep(SLEEP)
    
    # Load client models
    c_sets = [tf.keras.models.load_model(f, compile=False).get_weights()
              for f in r_models]
    
    # ABLATED BASELINE: Basic FedAvg - just average the weights
    # REMOVED: FedProx proximal correction
    # REMOVED: Dynamic μ computation
    avg = average_weight_sets(c_sets)
    
    # Save aggregated model
    tmpl.set_weights(avg)
    tmpl.save(f"global_weights/global_round{r}_baseline.keras")
    print(f"  Aggregated → global_round{r}_baseline.keras (Simple FedAvg)")
    
    # Compute and log global losses
    g_tr, g_va = compute_global_losses(r_metrics)
    log_global_losses(r, g_tr, g_va)

# Save final model
tmpl.save("final_global_baseline.keras")
print("\n" + "=" * 60)
print("Federated training complete → final_global_baseline.keras")
print("Used simple FedAvg (no FedProx)")
print("=" * 60)