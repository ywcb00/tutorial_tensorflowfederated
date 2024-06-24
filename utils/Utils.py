from model.ModelBuilderUtils import getMetrics

from prettytable import PrettyTable

class Utils:
    @classmethod
    def printEvaluations(self_class, eval_dict, config, first_col_name="Model"):
        metric_objects = getMetrics(config)
        eval_table = PrettyTable([first_col_name] + [mo.name for mo in metric_objects])
        for model_name, metrics in eval_dict.items():
            eval_table.add_row(
                [model_name] + [metrics[mo.name] for mo in metric_objects])
        return eval_table
