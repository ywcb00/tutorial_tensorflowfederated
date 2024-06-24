from model.ModelBuilderUtils import getMetrics

from prettytable import PrettyTable

class Utils:
    @classmethod
    def printEvaluations(self_class, eval_dict, config):
        metric_objects = getMetrics(config)
        eval_table = PrettyTable(["Model"] + [mo.name for mo in metric_objects])
        for model_name, metrics in eval_dict.items():
            for mo in metric_objects:
                eval_table.add_row(
                    [model_name] + [metrics[mo.name] for mo in metric_objects])
        print(eval_table)
