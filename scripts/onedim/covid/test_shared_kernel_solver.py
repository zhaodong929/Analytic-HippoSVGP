import argparse
import csv
import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch


def _prepare_module(module_name, seed, inducing_size, rff_sample_size, device):
    mod = importlib.import_module(module_name)
    mod = importlib.reload(mod)
    mod.args = SimpleNamespace(seed_num=seed)
    mod.seed_everything(seed)
    mod.rff_sample_size = rff_sample_size
    mod.hippo = mod.HiPPO_LegS(inducing_size, device)
    return mod


def _reset_module_outputs(base_dir, seed):
    model_dir = Path(base_dir) / f"models{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.csv").write_text("eval_task,after_task,rmse,r2,nlpd,ece\n")
    (model_dir / "training.log").write_text("")
    return model_dir

# Save the Task1 shared state checkpoint: 
# (kernel + likelihood + variational parameters + w + distance information).
def _save_task1_checkpoint(path, model, w, prev_discrete_steps, prev_end):
    checkpoint = {
        "kernel_state_dict": {k: v.detach().cpu() for k, v in model.kernel.state_dict().items()},
        "likelihood_state_dict": {
            k: v.detach().cpu() for k, v in model.likelihood.state_dict().items()
        },
        "mv": model.mv.detach().cpu(),
        "Lv": model.Lv.detach().cpu(),
        "w": w.detach().cpu(),
        "prev_discrete_steps": int(prev_discrete_steps),
        "prev_end": float(prev_end),
        "inducing_size": int(model.inducing_size),
        "num_inducing_old": int(model.num_inducing_old) if model.num_inducing_old is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def _load_task1_checkpoint(path):
    # Keep checkpoint tensors on CPU; move to target device explicitly where needed.
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def _apply_task1_checkpoint(prev_model, checkpoint, device):
    kernel_sd = {k: v.to(device=device) for k, v in checkpoint["kernel_state_dict"].items()}
    lik_sd = {k: v.to(device=device) for k, v in checkpoint["likelihood_state_dict"].items()}
    prev_model.kernel.load_state_dict(kernel_sd)
    prev_model.likelihood.load_state_dict(lik_sd)
    prev_model.mv = torch.nn.Parameter(checkpoint["mv"].to(device=device).clone(), requires_grad=True)
    prev_model.Lv = torch.nn.Parameter(checkpoint["Lv"].to(device=device).clone(), requires_grad=True)
    return prev_model


def _clone_task_cache(days_test_list, deaths_test_list, days_train, deaths_train):
    return (
        [x.clone().detach() for x in days_test_list],
        [x.clone().detach() for x in deaths_test_list],
        days_train.clone().detach(),
        deaths_train.clone().detach(),
    )

# After each task, a strong check is performed to ensure that the kernel/likelihood has not drifted (otherwise, an error will be reported).
def _assert_shared_state_unchanged(model, checkpoint, solver_name, task_id, atol=1e-12, rtol=1e-9):
    for name, ref_val in checkpoint["kernel_state_dict"].items():
        cur_val = model.kernel.state_dict()[name].detach().cpu()
        ref_val = ref_val.detach().cpu()
        if not torch.allclose(cur_val, ref_val, atol=atol, rtol=rtol):
            diff = torch.max(torch.abs(cur_val - ref_val)).item()
            raise RuntimeError(
                f"[{solver_name}] kernel parameter drift detected at task {task_id}: {name}, max_abs_diff={diff}"
            )
    for name, ref_val in checkpoint["likelihood_state_dict"].items():
        cur_val = model.likelihood.state_dict()[name].detach().cpu()
        ref_val = ref_val.detach().cpu()
        if not torch.allclose(cur_val, ref_val, atol=atol, rtol=rtol):
            diff = torch.max(torch.abs(cur_val - ref_val)).item()
            raise RuntimeError(
                f"[{solver_name}] likelihood parameter drift detected at task {task_id}: {name}, max_abs_diff={diff}"
            )


def _read_metrics(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _write_comparison_csv(path, ode_rows, analytic_rows, task_start, task_end):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["solver", "eval_task", "after_task", "rmse", "r2", "nlpd", "ece"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for solver, rows in [("ode", ode_rows), ("analytic", analytic_rows)]:
            for row in rows:
                after_task = int(row["after_task"])
                if task_start <= after_task <= task_end:
                    writer.writerow(
                        {
                            "solver": solver,
                            "eval_task": row["eval_task"],
                            "after_task": row["after_task"],
                            "rmse": row["rmse"],
                            "r2": row["r2"],
                            "nlpd": row["nlpd"],
                            "ece": row["ece"],
                        }
                    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Task 1 with ODE solver, save shared Task1 kernel/state, then run "
            "Task2-Task5 with ODE and analytic solvers using the exact same frozen Task1 state."
        )
    )
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--inducing_size", type=int, default=15)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--step_size", type=float, default=2.0)
    parser.add_argument("--rff_sample_size", type=int, default=1000)
    parser.add_argument("--task_start", type=int, default=2)
    parser.add_argument("--task_end", type=int, default=5)
    parser.add_argument(
        "--shared_dir",
        type=str,
        default="ohsgp_shared",
        help="Directory for shared Task1 checkpoint and merged comparison metrics.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="task1_shared_state.pt",
        help="Task1 shared checkpoint filename.",
    )
    args = parser.parse_args()
    if args.task_start < 2:
        raise ValueError("--task_start must be >= 2, because Task1 is trained only once with ODE.")
    if args.task_end < args.task_start:
        raise ValueError("--task_end must be >= --task_start.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ode_mod = _prepare_module(
        "scripts.onedim.covid.test_ohsgp",
        args.seed_num,
        args.inducing_size,
        args.rff_sample_size,
        device,
    )
    analytic_mod = _prepare_module(
        "scripts.onedim.covid.test_ohsgp_analytic",
        args.seed_num,
        args.inducing_size,
        args.rff_sample_size,
        device,
    )

    ode_model_dir = _reset_module_outputs("ohsgp", args.seed_num)
    analytic_model_dir = _reset_module_outputs("ohsgp_analytic", args.seed_num)
    
    #ODE only train Task1
    prev_model, days_test_list, deaths_test_list, days_train, deaths_train, w, prev_discrete_steps, prev_end = (
        ode_mod.train_task1(
            device=device,
            inducing_size=args.inducing_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            step_size=args.step_size,
        )
    )

    shared_dir = Path(args.shared_dir) / f"models{args.seed_num}"
    checkpoint_path = shared_dir / args.checkpoint_name
    _save_task1_checkpoint(checkpoint_path, prev_model, w, prev_discrete_steps, prev_end)

    # Tasks 2-5 both pipelines are initialized from the same checkpoint.
    checkpoint = _load_task1_checkpoint(checkpoint_path)

    ode_prev_model = _apply_task1_checkpoint(deepcopy(prev_model), checkpoint, device)
    analytic_prev_model = _apply_task1_checkpoint(deepcopy(prev_model), checkpoint, device)

    ode_days_test_list, ode_deaths_test_list, ode_days_train, ode_deaths_train = _clone_task_cache(
        days_test_list, deaths_test_list, days_train, deaths_train
    )
    ana_days_test_list, ana_deaths_test_list, ana_days_train, ana_deaths_train = _clone_task_cache(
        days_test_list, deaths_test_list, days_train, deaths_train
    )

    #Call the two existing train_nexttask sets (ODE / analytic) separately, and keep the shared parameters frozen for comparison.
    shared_w = checkpoint["w"].to(device=device).clone()
    ode_prev_discrete_steps = int(checkpoint["prev_discrete_steps"])
    analytic_prev_discrete_steps = int(checkpoint["prev_discrete_steps"])
    ode_prev_end = float(checkpoint["prev_end"])
    analytic_prev_end = float(checkpoint["prev_end"])

    for task_id in range(args.task_start, args.task_end + 1):
        ode_prev_model, ode_days_test_list, ode_deaths_test_list, ode_days_train, ode_deaths_train, shared_w, ode_prev_discrete_steps, ode_prev_end = (
            ode_mod.train_nexttask(
                ode_prev_model,
                ode_days_test_list,
                ode_deaths_test_list,
                ode_days_train,
                ode_deaths_train,
                device,
                task_id,
                args.inducing_size,
                args.num_epochs,
                args.lr,
                args.step_size,
                shared_w,
                ode_prev_discrete_steps,
                ode_prev_end,
            )
        )
        _assert_shared_state_unchanged(ode_prev_model, checkpoint, "ode", task_id)

        analytic_prev_model, ana_days_test_list, ana_deaths_test_list, ana_days_train, ana_deaths_train, shared_w, analytic_prev_discrete_steps, analytic_prev_end = (
            analytic_mod.train_nexttask(
                analytic_prev_model,
                ana_days_test_list,
                ana_deaths_test_list,
                ana_days_train,
                ana_deaths_train,
                device,
                task_id,
                args.inducing_size,
                args.num_epochs,
                args.lr,
                args.step_size,
                shared_w,
                analytic_prev_discrete_steps,
                analytic_prev_end,
            )
        )
        _assert_shared_state_unchanged(analytic_prev_model, checkpoint, "analytic", task_id)

    ode_rows = _read_metrics(ode_model_dir / "metrics.csv")
    analytic_rows = _read_metrics(analytic_model_dir / "metrics.csv")
    _write_comparison_csv(
        shared_dir / "metrics_solver_comparison.csv",
        ode_rows,
        analytic_rows,
        args.task_start,
        args.task_end,
    )

    print(f"Saved shared Task1 checkpoint: {checkpoint_path}")
    print(f"Saved merged Task{args.task_start}-Task{args.task_end} comparison: {shared_dir / 'metrics_solver_comparison.csv'}")


if __name__ == "__main__":
    main()
