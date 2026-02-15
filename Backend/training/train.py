import os
import sys
import time
from torch.utils.data import DataLoader, random_split

# Add parent directory to path to import app module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.model_factory import ModelFactory
from training.dataset import SkeletonDataset
from config import settings


def main():
    print(f"Starting training with model: {settings.ACTIVE_MODEL}")
    print(f"Data directory: {settings.DATA_DIR}")
    print(f"Device setting: {settings.DEVICE}")
    print(f"Epochs: {settings.EPOCHS}")
    print(f"Batch size: {settings.BATCH_SIZE}")
    print(f"Learning rate: {settings.LEARNING_RATE}")

    # Check if data directory exists
    if not os.path.exists(settings.DATA_DIR):
        print(f"ERROR: Data directory does not exist: {settings.DATA_DIR}")
        sys.exit(1)

    # Dataset
    try:
        print("Loading dataset...")
        dataset = SkeletonDataset(settings.DATA_DIR)
        print(f"Dataset loaded: {len(dataset)} samples found")
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if len(dataset) == 0:
        print(f"ERROR: No samples found in {settings.DATA_DIR}")
        sys.exit(1)

    # Print label distribution
    from collections import Counter
    label_counts = Counter(dataset.labels)
    print(f"Label distribution: { {k: v for k, v in sorted(label_counts.items())} }")
    for label, count in sorted(label_counts.items()):
        pct = 100.0 * count / len(dataset)
        print(f"  Label {label} ({'correct' if label == 0 else 'incorrect'}): {count} ({pct:.1f}%)")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=settings.BATCH_SIZE, num_workers=0)

    print(f"Train samples: {train_size}, Val samples: {val_size}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    try:
        print(f"Building model: {settings.ACTIVE_MODEL}")
        model = ModelFactory.create(settings.ACTIVE_MODEL)
        model.build()
        print(f"Model info: {model.get_model_info()}")
    except Exception as e:
        print(f"ERROR building model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    best_val_acc = 0.0
    os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(settings.WEIGHTS_DIR, f"{settings.ACTIVE_MODEL}_best.pt")

    print(f"\nStarting training for {settings.EPOCHS} epochs...")
    print(f"Weights will be saved to: {save_path}\n")

    total_start = time.time()
    try:
        for epoch in range(settings.EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = 0, 0
            num_train_batches = len(train_loader)
            for i, batch in enumerate(train_loader):
                try:
                    metrics = model.train_step(batch)
                    train_loss += metrics["loss"]
                    train_acc += metrics["accuracy"]
                    # Per-batch progress (overwrite line)
                    print(
                        f"\r  Training batch {i + 1}/{num_train_batches} "
                        f"| loss: {metrics['loss']:.4f} acc: {metrics['accuracy']:.4f}",
                        end="", flush=True
                    )
                except Exception as e:
                    print(f"\nERROR in training batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            print()  # newline after progress

            train_loss /= num_train_batches
            train_acc /= num_train_batches

            # Validate
            val_loss, val_acc = 0, 0
            num_val_batches = len(val_loader)
            for i, batch in enumerate(val_loader):
                try:
                    metrics = model.eval_step(batch)
                    val_loss += metrics["loss"]
                    val_acc += metrics["accuracy"]
                    print(
                        f"\r  Validating batch {i + 1}/{num_val_batches}",
                        end="", flush=True
                    )
                except Exception as e:
                    print(f"\nERROR in validation batch {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            print()  # newline after progress

            val_loss /= num_val_batches
            val_acc /= num_val_batches

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{settings.EPOCHS} "
                f"({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_weights(save_path)
                print(f"  → Saved best model (val_acc={val_acc:.4f})")

            # ETA estimate
            elapsed = time.time() - total_start
            epochs_done = epoch + 1
            avg_epoch_time = elapsed / epochs_done
            remaining = avg_epoch_time * (settings.EPOCHS - epochs_done)
            print(f"  ETA: {remaining / 60:.1f} min remaining\n")

        total_time = time.time() - total_start
        print(f"\nTraining complete in {total_time / 60:.1f} min. Best val accuracy: {best_val_acc:.4f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()