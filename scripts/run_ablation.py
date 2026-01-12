# scripts/run_ablation.py
"""
示例：
python scripts/run_ablation.py --data_root ./data --epochs 20 --batch_size 128 --glove_path ./embeddings/glove.6B.300d.txt
"""
import argparse, subprocess, os, sys, itertools

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--glove_path", type=str, default="./embeddings/glove.6B.300d.txt")
    ap.add_argument("--lambda_attr", type=float, default=0.5)
    ap.add_argument("--lambda_l1", type=float, default=1e-4)
    ap.add_argument("--log_dir", type=str, default="./outputs/logs")
    args = ap.parse_args()

    combos = [
        # (run_name, kg_init, use_attr, lambda_attr, lambda_l1, use_kg)
        ("kgxnn_rand_noattr",  "random", False, 0.0, args.lambda_l1, True),
        ("kgxnn_glove_noattr", "glove",  False, 0.0, args.lambda_l1, True),
        ("kgxnn_glove_attr",   "glove",  True,  args.lambda_attr, args.lambda_l1, True),
        ("kgxnn_glove_frozen_attr", "glove_frozen", True, args.lambda_attr, args.lambda_l1, True),
        ("kgxnn_glove_attr_noL1",   "glove", True, args.lambda_attr, 0.0, True),
        ("kgxnn_noKG", "glove", True, args.lambda_attr, args.lambda_l1, False),
    ]

    py = sys.executable
    for run_name, kg_init, use_attr, la, l1, use_kg in combos:
        cmd = [
            py, "train_kgxnn.py",
            "--data_root", args.data_root,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--img_size", str(args.img_size),
            "--kg_init", kg_init,
            "--lambda_attr", str(la),
            "--lambda_l1", str(l1),
            "--log_dir", args.log_dir,
            "--run_name", run_name,
            "--use_kg", str(use_kg),
        ]
        if kg_init in ("glove", "glove_frozen"):
            cmd += ["--glove_path", args.glove_path]
        if use_attr:
            cmd += ["--use_attr_supervision"]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)
