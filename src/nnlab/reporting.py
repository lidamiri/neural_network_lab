def _format_matrix(matrix, labels):
    width = max(5, max(len(str(label)) for label in labels) + 1)
    header = " " * width + "".join(f"{str(label):>{width}}" for label in labels)
    rows = [header]
    for label, row in zip(labels, matrix):
        rows.append(f"{str(label):>{width}}" + "".join(f"{int(value):>{width}}" for value in row))
    return "\n".join(rows)


def classification_report(metrics, class_names=None):
    matrix = metrics["confusion_matrix"]
    labels = class_names or list(range(len(matrix)))
    lines = [
        f"accuracy        {metrics['accuracy']:.4f}",
        f"macro precision {metrics['macro_precision']:.4f}",
        f"macro recall    {metrics['macro_recall']:.4f}",
        f"macro f1        {metrics['macro_f1']:.4f}",
    ]
    if "top2_accuracy" in metrics:
        lines.append(f"top-2 accuracy  {metrics['top2_accuracy']:.4f}")
    lines.append("")
    lines.append(f"{'class':<12}{'precision':>12}{'recall':>12}{'f1':>12}")
    for index, label in enumerate(labels):
        lines.append(
            f"{str(label):<12}"
            f"{metrics['per_class_precision'][index]:>12.4f}"
            f"{metrics['per_class_recall'][index]:>12.4f}"
            f"{metrics['per_class_f1'][index]:>12.4f}"
        )
    lines.append("")
    lines.append("confusion matrix (rows: actual, columns: predicted)")
    lines.append(_format_matrix(matrix, labels))
    return "\n".join(lines)


def regression_report(metrics):
    return "\n".join(
        [
            f"mse   {metrics['mse']:.6f}",
            f"rmse  {metrics['rmse']:.6f}",
            f"mae   {metrics['mae']:.6f}",
            f"r2    {metrics['r2']:.4f}",
        ]
    )


def benchmark_table(rows):
    if not rows:
        return "no runs were executed"
    columns = list(rows[0].keys())
    widths = {column: max(len(column), max(len(_cell(row[column])) for row in rows)) for column in columns}
    header = "  ".join(f"{column:<{widths[column]}}" for column in columns)
    rule = "  ".join("-" * widths[column] for column in columns)
    body = ["  ".join(f"{_cell(row[column]):<{widths[column]}}" for column in columns) for row in rows]
    return "\n".join([header, rule] + body)


def _cell(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
