from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.models import yolo
from ultralytics.utils.metrics import ConfusionMatrix, ClassifyMetrics
from ultralytics.models.yolo.classify import ClassificationValidator, ClassificationTrainer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics.utils import plt_settings


@plt_settings()
def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None):
    import pandas as pd  # scope for faster 'import ultralytics'
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(3, 2, figsize=(6, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(
        files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem,
                           linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":",
                           label="smooth", linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in {8, 9, 10}:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")
    ax[1].legend()
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


class ModifiedClassifyMetrics(ClassifyMetrics):
    def __init__(self) -> None:
        super().__init__()
        self.precision = 0
        self.recall = 0

    def set_precision_recall(self, precision, recall):
        self.precision = precision
        self.recall = recall

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5,  self.precision, self.recall, self.fitness]))

    @property
    def keys(self):
        return ["metrics/accuracy_top1", "metrics/accuracy_top5", "metrics/precision", "metrics/recall"]


class ModifiedConfusionMatrix(ConfusionMatrix):
    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        super().__init__(nc, conf, iou_thres, task)

    def tp_fp_fn(self):
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        fn = self.matrix.sum(0) - tp
        return tp, fp, fn

    def precision(self):
        tp, fp, _ = self.tp_fp_fn()
        precision = tp / (tp + fp + 1e-9)
        return precision

    def recall(self):
        tp, _, fn = self.tp_fp_fn()
        recall = tp / (tp + fn + 1e-9)
        return recall


class ModifiedClassificationValidator(ClassificationValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.precision = 0
        self.recall = 0
        self.metrics = ModifiedClassifyMetrics()

    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ModifiedConfusionMatrix(
            nc=self.nc, conf=self.args.conf, task="classify")
        self.pred = []
        self.targets = []

    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics()
        precision = self.confusion_matrix.precision()
        recall = self.confusion_matrix.recall()

        overall_precision = np.mean(precision)
        overall_recall = np.mean(recall)

        self.metrics.set_precision_recall(overall_precision, overall_recall)

    def get_stats(self):
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def print_results(self):
        pf = "%22s%11.3g, %11.3g, %.5f, %.5f"
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5,
                    self.metrics.precision, self.metrics.recall))


class ModifiedClassificationTrainer(ClassificationTrainer):
    def get_validator(self):
        self.loss_names = ["loss"]
        return ModifiedClassificationValidator(self.test_loader, self.save_dir, _callbacks=self.callbacks)

    def plot_metrics(self):
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)


class ModifiedYolo(YOLO):

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": ModifiedClassificationTrainer,
                "validator": ModifiedClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
        }
