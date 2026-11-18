# Generates the Q7 test-set submission files from the best fine-tuned checkpoint.
# Run from the part2 folder:  PYTHONPATH=. python3 run_test.py
import load_data, t5_utils, train_t5

class A:
    finetune = True
    experiment_name = "experiment_v2"

model = t5_utils.load_model_from_checkpoint(A(), best=True)
test_loader = load_data.get_dataloader(16, "test")
train_t5.test_inference(A(), model, test_loader,
                        "results/t5_ft_experiment_test.sql",
                        "records/t5_ft_experiment_test.pkl")
print("DONE")
