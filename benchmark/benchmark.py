import pickle
from torch.utils.benchmark import Timer, Compare
from .config import BenchmarkConfig
from transformers import AutoModel


def benchmark_model(
        model,
        inputs,
        run_func,
        label='',
        sub_label='',
        description='',
        num_threads=1,
        min_run_time=0,
        num_run_times=None
):
    """
    Benchmark a model on inputs through a run_func

    Args:
        model (torch.nn.Module): pytorch model
        inputs (dict): model inputs
        run_func (function): function to run the model or session in a specific way
        label (str): label for the benchmark
        sub_label (str): sub label for the benchmark
        description (str): description for the benchmark
        num_threads (int): number of threads to use for the benchmark
        min_run_time (float): minimum run time for the benchmark
        num_run_times (int): number of times to run the benchmark

    Returns:
        measurement (torch.utils.benchmark.Measurement): benchmark measurement
    """

    timer = Timer(
        stmt=f"run_func(model, inputs)",
        globals={'run_func': run_func,
                 'model': model,
                 'inputs': inputs},

        label=label,
        sub_label=sub_label,
        description=description,

        num_threads=num_threads,
    )

    if num_run_times is not None:
        measurement = timer.timeit(number=num_run_times)
    else:
        measurement = timer.blocked_autorange(min_run_time=min_run_time)

    return measurement


class PytorchBenchmark:
    """
    Benchmark a PyTorch model on a space of configurations
    """
    
    benchmark_label = 'Pytorch Benchmark'
    
    def __init__(self, config=BenchmarkConfig()):
        
        self.config = config

    def run(self):
        
        # prepare some variables
        self.benchmark_results = dict()

        for model_name in self.config.models_space:
            model = AutoModel.from_pretrained(model_name)

            for bs, sl, sp in self.config.inputs_space:
                sub_label = f'batch_size={bs}, seq_len={sl}, sparsity={sp}'
                inputs = self.config.dummy_inputs_func(bs, sl, sp)
            
                for device, th in self.config.env_space:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    model.to(device)

                    results = benchmark_model(
                            model=model,
                            inputs=inputs,

                            run_func=self.config.run_func,
                            min_run_time=self.config.min_run_time,
                            num_run_times=self.config.num_run_times,

                            label=self.benchmark_label,
                            description=f"{model_name} ({device})",
                            sub_label=sub_label,
                            num_threads=th,
                        )
                    
                    self.benchmark_results[(model_name, bs, sl, sp, th, device)] = results

    def print(self, trim_significant_figures=False, colorize=False):
        comparison = Compare(self.benchmark_results.values())

        if trim_significant_figures:
            comparison.trim_significant_figures()

        if colorize:
            comparison.colorize(rowwise=True)

        comparison.print()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.benchmark_results, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.benchmark_results = pickle.load(f)
