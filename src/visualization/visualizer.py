import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import learning_curve
from typing import Dict, Any, List

from src.config import config

class Visualizer:
    @staticmethod
    def plot_boxplot(X: pd.DataFrame, target_name: str):
        # Academic and Publication Ready Style
        sns.set_theme(style="ticks", context="paper", font="serif")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        flierprops = dict(marker='o', markerfacecolor='black', markersize=4, alpha=0.6, markeredgecolor='none')
        medianprops = dict(color="black", linewidth=1.5)
        
        sns.boxplot(
            data=X, 
            color="#E0E0E0", # Light gray for a professional academic look
            width=0.5,
            linewidth=1.2,
            flierprops=flierprops,
            medianprops=medianprops, # Black median line
            ax=ax
        )
        
        # Professional formatting
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylabel("Values", fontsize=12, fontweight='bold')
        ax.set_xlabel("Features", fontsize=12, fontweight='bold')
        
        # Grid lines behind the boxes for better readability
        ax.yaxis.grid(True, linestyle='--', color='#b0b0b0', alpha=0.7)
        ax.set_axisbelow(True)
        
        sns.despine()
        plt.tight_layout()
        
        # Save with high DPI for publication (600 dpi)
        plt.savefig(config.output_path(f"boxplot_selected_features_{target_name}.png"), dpi=600, format='png', bbox_inches='tight')
        plt.close()
        sns.reset_orig()
        print(f"[{target_name}] Publication-ready Box-Plot saved.")

    @staticmethod
    def plot_scatter_all(results: Dict[str, Dict[str, float]], test_preds: Dict[str, np.ndarray], y_test: pd.Series, target_name: str):
        all_5_models = sorted(results.keys(), key=lambda k: results[k]['CV_RMSE_Mean'])
        
        fig = plt.figure(figsize=(14, 18))
        fig.suptitle(f'{target_name} - Actual vs Predicted on Test Set', fontsize=16, fontweight='bold', color='#333333')
        
        gs = fig.add_gridspec(3, 4)
        axes = [
            fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
            fig.add_subplot(gs[2, 1:3])
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for ax, m_name, color in zip(axes, all_5_models, colors):
            y_m_pred = test_preds[m_name]
            
            ax.scatter(y_test, y_m_pred, alpha=0.65, edgecolors='w', linewidth=0.5, color=color, s=50)
            
            min_val = min(y_test.min(), y_m_pred.min())
            max_val = max(y_test.max(), y_m_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7, label='Ideal Line (Y=X)')
            
            m_r2 = results[m_name]['CV_R2_Mean']
            m_rmse = results[m_name]['CV_RMSE_Mean']
            textstr = f"$R^2$: {m_r2:.4f}\nRMSE: {m_rmse:.4f}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='silver'))
                    
            ax.set_title(f'{m_name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Actual Values', fontsize=11)
            ax.set_ylabel('Predicted Values', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='lower right', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(config.output_path(f'scatter_plot_all5_{target_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_prediction_curve(y_test: pd.Series, y_pred: np.ndarray, target_name: str, model_name: str):
        plt.figure(figsize=(12, 5))
        plt.plot(y_test.values, label='Actual Value', color='#1f77b4', alpha=0.8, linewidth=1.5)
        plt.plot(y_pred, label='Predicted Value', color='#ff7f0e', alpha=0.8, linestyle='--', linewidth=1.5)
        plt.xlabel('Sample Index (Test Set)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'{target_name} - Actual vs Predicted Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(config.output_path(f'prediction_curve_{target_name}.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_learning_curve(model: Any, X_train: pd.DataFrame, y_train: pd.Series, target_name: str, model_name: str):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='neg_mean_squared_error'
        )
        
        train_rmse = np.sqrt(-train_scores)
        test_rmse = np.sqrt(-test_scores)
        
        train_mean = np.mean(train_rmse, axis=1)
        train_std = np.std(train_rmse, axis=1)
        test_mean = np.mean(test_rmse, axis=1)
        test_std = np.std(test_rmse, axis=1)

        plt.figure(figsize=(10, 6))
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#ff7f0e")
        
        plt.plot(train_sizes, train_mean, 'o-', color="#1f77b4", linewidth=2.5, markersize=8, label="Training Error (RMSE)")
        plt.plot(train_sizes, test_mean, 's--', color="#ff7f0e", linewidth=2.5, markersize=8, label="Cross-validation Error (RMSE)")
        
        plt.xlabel("Training Set Size (Number of Samples)", fontsize=12)
        plt.ylabel("RMSE Score", fontsize=12)
        plt.title(f'{target_name} - Learning Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(config.output_path(f'learning_curve_{target_name}.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_shap(model: Any, X_train: pd.DataFrame, target_name: str, model_name: str):
        print(f"[{target_name}] Generating SHAP plots...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        except Exception as e:
            print(f"[{target_name}] TreeExplainer failed: {e}. Trying KernelExplainer...")
            X_sample = shap.sample(X_train, 100) if len(X_train) > 100 else X_train
            predict_fn = lambda x: model.predict(x)
            explainer = shap.KernelExplainer(predict_fn, X_sample)
            shap_values = explainer.shap_values(X_train)

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title(f"SHAP Summary Plot - {target_name} ({model_name})")
        plt.tight_layout()
        plt.savefig(config.output_path(f"shap_summary_{target_name}.png"))
        plt.close()

        # SHAP Bar Plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {target_name} ({model_name})")
        plt.tight_layout()
        plt.savefig(config.output_path(f"shap_bar_{target_name}.png"))
        plt.close()
