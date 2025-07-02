
# SPDX-License-Identifier: MIT
# © 2025 Siamak Solat
"""
federated_server.py

  Workflow:
  1. Each client (one per zone) trains a local model for a given round r
     and uploads:
          •  <zone>_round{r}.keras           – model weights
          •  <zone>_metrics_round{r}.json    – loss, sample counts
  2. The server polls until all zones have uploaded their files.
  3. It averages the weights and then applies a FedProx-style proximal correction
           w ← w – μ · (w – w_prev)
     where μ is computed dynamically from the inter-client dispersion and
     clipped to the range [MU_MIN, MU_MAX].
  4. Global losses are recomputed from the client-reported metrics and
     appended to global_metrics.csv.
  5. The aggregated weights are saved to
         global_weights/global_round{r}.keras
     and the loop continues for the configured number of rounds.

  After the final round the model is exported as final_global.keras.
"""

import os, time, yaml, json
import numpy as np
import tensorflow as tf
import absl.logging as absl_logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

CFG = yaml.safe_load(open("all_zones.yaml"))

ROUNDS  = CFG["common"]["federated"]["rounds"]
ZONES   = list(CFG["zones"].keys())
MU_BASE = CFG["common"]["federated"].get("mu_base", 0.01)
MU_MIN  = CFG["common"]["federated"].get("mu_min", 0.005)
MU_MAX  = CFG["common"]["federated"].get("mu_max", 0.05)
SLEEP   = 10

os.makedirs("global_weights", exist_ok=True)
if not os.path.exists("global_metrics.csv"):
    with open("global_metrics.csv", "w") as f:
        f.write("round,train_loss,val_loss,mu\n")

def all_exist(paths):             return all(os.path.exists(p) for p in paths)
def average_weight_sets(wsets):   return [np.mean([w[i] for w in wsets], 0)
                                          for i in range(len(wsets[0]))]

def compute_dispersion(wsets, avg):
    d = [np.linalg.norm(np.concatenate([(w-a).ravel()
          for w, a in zip(ws, avg)])) for ws in wsets]
    return float(np.mean(d))

def dynamic_mu(wsets, avg):
    mu = MU_BASE * compute_dispersion(wsets, avg)
    return min(max(mu, MU_MIN), MU_MAX)

def compute_global_losses(metric_paths):
    tot_tr = tot_va = n_tr = n_va = 0
    for p in metric_paths:
        with open(p) as f: m = json.load(f)
        tot_tr += m["train_loss"] * m["n_train"]
        tot_va += m["val_loss"]   * m["n_val"]
        n_tr   += m["n_train"];  n_va += m["n_val"]
    return (tot_tr/n_tr if n_tr else float("nan"),
            tot_va/n_va if n_va else float("nan"))

def log_global_losses(rnd, g_tr, g_va, mu):
    print(f"Global losses | round {rnd}: "
          f"train={g_tr:.4f}  val={g_va:.4f}  μ={mu:.4f}")
    with open("global_metrics.csv","a") as f:
        f.write(f"{rnd},{g_tr},{g_va},{mu}\n")

# ───────── Round 0 ─────────
print("Server waiting for all clients to finish round 0 …")
r0_models  = [f"{z}_round0.keras"        for z in ZONES]
r0_metrics = [f"{z}_metrics_round0.json" for z in ZONES]
while not all_exist(r0_models + r0_metrics):
    time.sleep(SLEEP)

print("All round-0 files detected. Aggregating …")
w_sets = [tf.keras.models.load_model(f, compile=False).get_weights()
          for f in r0_models]
avg = average_weight_sets(w_sets)
tmpl = tf.keras.models.load_model(r0_models[0], compile=False)
tmpl.set_weights(avg)
tmpl.save("global_weights/global_round0.keras")
g_tr, g_va = compute_global_losses(r0_metrics)
log_global_losses(0, g_tr, g_va, 0.0)

# ───────── Federated rounds 1..R ─────────
for r in range(1, ROUNDS+1):
    print(f"\n Round {r}/{ROUNDS} – waiting for uploads …")
    r_models  = [f"{z}_round{r}.keras"        for z in ZONES]
    r_metrics = [f"{z}_metrics_round{r}.json" for z in ZONES]
    while not all_exist(r_models + r_metrics):
        time.sleep(SLEEP)

    c_sets = [tf.keras.models.load_model(f, compile=False).get_weights()
              for f in r_models]
    avg = average_weight_sets(c_sets)

    mu_val = dynamic_mu(c_sets, avg)
    prev_w = tf.keras.models.load_model(
        f"global_weights/global_round{r-1}.keras",
        compile=False).get_weights()
    for i in range(len(avg)):
        avg[i] -= mu_val * (avg[i] - prev_w[i])

    tmpl.set_weights(avg)
    tmpl.save(f"global_weights/global_round{r}.keras")
    print(f"  Aggregated → global_round{r}.keras (μ={mu_val:.4f})")

    g_tr, g_va = compute_global_losses(r_metrics)
    log_global_losses(r, g_tr, g_va, mu_val)

tmpl.save("final_global.keras")
print("\n  Federated training complete → final_global.keras")
