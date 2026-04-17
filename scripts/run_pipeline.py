"""
ForzaTek AI — Pipeline Orchestrator
====================================
One-shot script: train the model, run predictions on unlabeled frames,
queue the most uncertain for review.

Usage:
  python -m scripts.run_pipeline --round 1 --epochs 30
  python -m scripts.run_pipeline --round 2 --epochs 20 --resume models/round_1.pt

The typical active-learning loop:
  1. Label 500 frames manually via the /label web UI
  2. Run:  python -m scripts.run_pipeline --round 1
  3. Open /label — top 50 most uncertain frames are queued
  4. Review them (accept / fix / reject)
  5. Run:  python -m scripts.run_pipeline --round 2 --resume models/round_1.pt
  6. Repeat until you're happy with the model's accuracy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import train as run_training
from training.predict import run_predictions
from backend.database import init_db, DB_PATH, get_active_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True,
                    help="Round number (1, 2, 3, ...)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resume", type=str, default=None,
                    help="Checkpoint to fine-tune from (for rounds 2+)")
    ap.add_argument("--predict-limit", type=int, default=500,
                    help="How many unlabeled frames to run predictions on")
    ap.add_argument("--queue-top-k", type=int, default=50,
                    help="How many uncertain frames to queue for review")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training, only run prediction with active model")
    ap.add_argument("--skip-predict", action="store_true",
                    help="Train only, don't generate proposals")
    args = ap.parse_args()

    init_db(DB_PATH)

    model_id = None
    if not args.skip_train:
        print(f"\n{'='*60}\n  ROUND {args.round} — TRAINING\n{'='*60}")
        model_id, best = run_training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume=args.resume,
            round_num=args.round,
        )
        print(f"\nTraining done. Best val mIoU: {best:.3f}")

    if not args.skip_predict:
        print(f"\n{'='*60}\n  ROUND {args.round} — PREDICTION\n{'='*60}")
        active = get_active_model()
        if not active:
            print("No active model — cannot run predictions")
            return
        run_predictions(
            model_path=active["path"],
            limit=args.predict_limit,
            queue_top_k=args.queue_top_k,
            round_num=args.round,
            model_id=active["id"],
        )
        print("\nPrediction done.")
        print(f"\nNext step: open http://localhost:8000/label and review "
              f"the {args.queue_top_k} queued frames.")


if __name__ == "__main__":
    main()
