import torch
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from tqdm import tqdm
from pathlib import Path    

from AD.visualization import plot_confusion_matrix, plot_roc_curves, plot_train_val_history, plot_class_wise_performance_over_all_phases, plot_average_performance_over_all_phases, plot_latent_space_umap, plot_latent_space_tsne

class Tester:
    
    def setup_testing(self, model_dir, device):

        self.model_dir = model_dir
        self.one_hot_encoder = joblib.load(f"{self.model_dir}/encoder.pkl")
        self.device = device

    def create_loss_history_plot(self):

        # Load the train and validation loss history
        train_loss_history = np.load(f"{self.model_dir}/train_loss_history.npy")
        val_loss_history = np.load(f"{self.model_dir}/val_loss_history.npy")
        
        # Save the plot
        file_name = f"{self.model_dir}/loss.pdf"
        plot_train_val_history(train_loss_history, val_loss_history, file_name)

    def create_metric_phase_plots(self):
        
        for metric in ['f1-score','precision','recall']:
            df = self.get_metric_over_all_phases(metric)
            plot_class_wise_performance_over_all_phases(metric, df, self.model_dir)
            plot_average_performance_over_all_phases(metric, df, self.model_dir)

    def create_classification_report(self, y_true, y_pred, file_name=None):
        
        # Only keep source where a true label exists
        idx = np.where(y_true!=None)[0]
        y_true = y_true[idx]
        y_pred = y_pred[idx]

        report = classification_report(y_true, y_pred)

        if file_name:
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            pd.DataFrame(report_dict).transpose().to_csv(file_name, index_label='Class')
        return report

    def get_metric_over_all_phases(self, metric):
        
        # Make sure metric is valid
        assert metric in ['f1-score','precision','recall']

        reports_dir = Path(f"{self.model_dir}/reports/")
        reports = list(reports_dir.glob("*.csv"))
        reports.sort()
        
        day_wise_metrics = []

        for report in reports:

            day = int(str(report).split("+")[1].split('.')[0])
            df = pd.read_csv(report, index_col='Class')[[metric]]
            df = df.rename(columns={metric:day})
            day_wise_metrics.append(df)

        combined_df = pd.concat(day_wise_metrics, axis=1, join='inner')
        combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)
        return combined_df

    def run_all_analysis(self, test_loader, d):

        self.eval()

        true_classes = []
        combined_pred_df = []
        combined_true_df = []
        combined_embeddings = []

        print(f'==========\nStarting Analysis for Trigger + {d} days...')

        # Run inference on the test set and combine the output dataframes
        for batch in tqdm(test_loader, desc='Testing'):

            # Move everything to the device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Run inference and get the predictions df
            pred_df = self.predict_conditional_probabilities_df(batch)
            embeddings = pd.DataFrame(self.get_latent_space_embeddings(batch).detach())

            # Make dataframe for true labels
            true_df = self.one_hot_encoder.transform(np.asarray(batch['label']).reshape(-1, 1)).toarray()
            true_df = pd.DataFrame(true_df, columns=pred_df.columns)

            # Maintain list across all batches
            true_classes += batch['label'].tolist()
            combined_pred_df.append(pred_df)
            combined_true_df.append(true_df)
            combined_embeddings.append(embeddings)
        
        true_classes = np.array(true_classes)
        combined_pred_df = pd.concat(combined_pred_df, ignore_index=True)
        combined_true_df = pd.concat(combined_true_df, ignore_index=True)
        combined_embeddings = pd.concat(combined_embeddings, ignore_index=True)

        # Make dirs for plots and reports
        Path(f"{self.model_dir}/plots/").mkdir(parents=True, exist_ok=True)
        Path(f"{self.model_dir}/reports/").mkdir(parents=True, exist_ok=True)

        # Get all the nodes at depth 
        classes = np.asarray(self.one_hot_encoder.categories_[0])
        pred_classes = classes[np.argmax(combined_pred_df.to_numpy(), axis=1)]
        true_classes = classes[np.argmax(combined_true_df.to_numpy(), axis=1)]
        
        title = f"Trigger+{d} days"

        # Make the recall confusion matrix plot
        Path(f"{self.model_dir}/plots/cf_recall").mkdir(parents=True, exist_ok=True)
        cf_img_file = f"{self.model_dir}/plots/cf_recall/cf_recall_trigger+{d}.pdf"
        plot_confusion_matrix(np.array(true_classes), np.array(pred_classes), classes, "true", title=title, file=cf_img_file)

        # Make the precision confusion matrix plot
        Path(f"{self.model_dir}/plots/cf_precision").mkdir(parents=True, exist_ok=True)
        cf_img_file = f"{self.model_dir}/plots/cf_precision/cf_precision_trigger+{d}.pdf"
        plot_confusion_matrix(np.array(true_classes), np.array(pred_classes), classes, "pred", title=title, file=cf_img_file)

        # Make the umap plots
        Path(f"{self.model_dir}/plots/umap").mkdir(parents=True, exist_ok=True)
        umap_img_file = f"{self.model_dir}/plots/umap/umap_trigger+{d}.pdf"
        plot_latent_space_umap(combined_embeddings, true_classes, title, umap_img_file)

        # Make the tsne plots
        Path(f"{self.model_dir}/plots/tsne").mkdir(parents=True, exist_ok=True)
        tsne_img_file = f"{self.model_dir}/plots/tsne/tsne_trigger+{d}.pdf"
        plot_latent_space_tsne(combined_embeddings, true_classes, title, tsne_img_file)

        # Make the ROC plot
        Path(f"{self.model_dir}/plots/roc").mkdir(parents=True, exist_ok=True)
        roc_img_file = f"{self.model_dir}/plots/roc/roc_trigger+{d}.pdf"
        plot_roc_curves(combined_true_df.to_numpy(), combined_pred_df.to_numpy(), classes, title=title, file=roc_img_file)

        # Make classification report
        report_file = f"{self.model_dir}/reports/report_trigger+{d}.csv"
        report = self.create_classification_report(np.array(true_classes), np.array(pred_classes), report_file)
        print(report)








