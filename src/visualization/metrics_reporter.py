import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

class MetricsReporter:
    def __init__(self, output_dir: str = 'results', dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.metrics_data: List[Dict[str, Any]] = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_metrics(
        self,
        noise_type: str,
        noise_intensity: float,
        n_iterations: int,
        elapsed_time: float,
        inertia: float,
        mse: float,
        psnr: float,
        precision: float,
        recall: float,
        f1: float
    ) -> None:
        self.metrics_data.append({
            'Noise Type': noise_type,
            'Noise Intensity': noise_intensity,
            'Iterations': n_iterations,
            'Time (s)': elapsed_time,
            'Inertia': inertia,
            'MSE': mse,
            'PSNR (dB)': psnr,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    
    def save_to_excel(self, filename: str = 'metrics_report.xlsx') -> str:
        filepath = self.output_dir / filename
        
        df = pd.DataFrame(self.metrics_data)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Metrics', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Metrics']
            
            for column in df:
                column_length = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_length + 2
        
        return str(filepath)
    
    def create_charts(self, filename: str = 'metrics_charts.png') -> str:
        if not self.metrics_data:
            raise ValueError("No metrics data available. Call add_metrics() first.")
        
        df = pd.DataFrame(self.metrics_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('K-Means Clustering Metrics Comparison', fontsize=16, fontweight='bold')
        
        noise_types = df['Noise Type'].unique()
        colors = plt.cm.Set2(range(len(noise_types)))
        color_map = dict(zip(noise_types, colors))
        
        for noise_type in noise_types:
            data = df[df['Noise Type'] == noise_type]
            color = color_map[noise_type]
            
            axes[0, 0].plot(data['Noise Intensity'], data['MSE'], 
                           marker='o', label=noise_type, color=color, linewidth=2)
            axes[0, 1].plot(data['Noise Intensity'], data['PSNR (dB)'], 
                           marker='o', label=noise_type, color=color, linewidth=2)
            axes[0, 2].plot(data['Noise Intensity'], data['F1 Score'], 
                           marker='o', label=noise_type, color=color, linewidth=2)
            axes[1, 0].plot(data['Noise Intensity'], data['Precision'], 
                           marker='o', label=noise_type, color=color, linewidth=2)
            axes[1, 1].plot(data['Noise Intensity'], data['Recall'], 
                           marker='o', label=noise_type, color=color, linewidth=2)
            axes[1, 2].plot(data['Noise Intensity'], data['Time (s)'], 
                           marker='o', label=noise_type, color=color, linewidth=2)
        
        axes[0, 0].set_xlabel('Noise Intensity')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_title('Mean Squared Error')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_xlabel('Noise Intensity')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].set_xlabel('Noise Intensity')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].set_xlabel('Noise Intensity')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_xlabel('Noise Intensity')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].set_xlabel('Noise Intensity')
        axes[1, 2].set_ylabel('Time (s)')
        axes[1, 2].set_title('Clustering Time')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def save_report(self, excel_filename: str = 'metrics_report.xlsx', 
                    chart_filename: str = 'metrics_charts.png') -> tuple[str, str]:
        excel_path = self.save_to_excel(excel_filename)
        chart_path = self.create_charts(chart_filename)
        return excel_path, chart_path
