"""
Visualization script
Visualizes evaluation results and creates comparison table
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))


class ResultsVisualizer:
    """Results visualization class"""

    def __init__(self, metrics_dir: str = "./results/metrics"):
        self.metrics_dir = metrics_dir
        self.results = {}

    def load_results(self):
        """Loads all evaluation results"""
        print("Loading evaluation results")

        # Base model results
        base_results_file = os.path.join(self.metrics_dir, "base_model_results.json")
        if os.path.exists(base_results_file):
            with open(base_results_file, "r") as f:
                self.results["base"] = json.load(f)
            print("Base model results loaded")

        # QLoRA results
        qlora_results_file = os.path.join(self.metrics_dir, "qlora_evaluation_results.json")
        if os.path.exists(qlora_results_file):
            with open(qlora_results_file, "r") as f:
                self.results["qlora"] = json.load(f)
            print("QLoRA results loaded")

        # GaLore results
        galore_results_file = os.path.join(self.metrics_dir, "galore_evaluation_results.json")
        if os.path.exists(galore_results_file):
            with open(galore_results_file, "r") as f:
                self.results["galore"] = json.load(f)
            print("GaLore results loaded")

    def create_comparison_table(self):
        """Creates comparison table according to assessment requirements"""
        print("\nCreating comparison table")

        if "base" not in self.results:
            print("Base model results not found")
            return None

        baseline_bleu = self.results["base"].get("bleu_4", 0)
        baseline_rouge = self.results["base"].get("rouge_l", 0)

        data = []

        for technique in ["qlora", "galore"]:
            technique_name = "QLoRA" if technique == "qlora" else "GaLore"

            if technique in self.results:
                result = self.results[technique]
                row = {
                    "Technique": technique_name,
                    "BLEU-4 (Before)": f"{baseline_bleu:.4f}",
                    "BLEU-4 (After)": f"{result.get('bleu_4', 0):.4f}",
                    "ROUGE-L (Before)": f"{baseline_rouge:.4f}",
                    "ROUGE-L (After)": f"{result.get('rouge_l', 0):.4f}",
                    "Peak Memory (GB)": f"{result.get('peak_memory_gb', 0):.2f}" if result.get('peak_memory_gb') else "N/A",
                    "Training Time (Hrs)": f"{result.get('training_time_hours', 0):.2f}" if result.get('training_time_hours') else "N/A",
                }
            else:
                row = {
                    "Technique": technique_name,
                    "BLEU-4 (Before)": f"{baseline_bleu:.4f}",
                    "BLEU-4 (After)": "N/A",
                    "ROUGE-L (Before)": f"{baseline_rouge:.4f}",
                    "ROUGE-L (After)": "N/A",
                    "Peak Memory (GB)": "N/A",
                    "Training Time (Hrs)": "N/A",
                }

            data.append(row)

        df = pd.DataFrame(data)

        print("\n" + "="*120)
        print("Comparison Table")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)

        # Save as CSV
        table_file = os.path.join(self.metrics_dir, "comparison_table.csv")
        df.to_csv(table_file, index=False)
        print(f"\nTable saved: {table_file}")

        return df

    def plot_bleu_rouge_comparison(self, output_dir: str = "./results/plots"):
        """BLEU and ROUGE comparison graph"""
        os.makedirs(output_dir, exist_ok=True)

        if not self.results:
            print("No results found")
            return

        techniques = []
        bleu_scores = []
        rouge_scores = []

        # Base model
        if "base" in self.results:
            techniques.append("Base Model")
            bleu_scores.append(self.results["base"].get("bleu_4", 0))
            rouge_scores.append(self.results["base"].get("rouge_l", 0))

        # Fine-tuned models
        for key, name in [("qlora", "QLoRA"), ("galore", "GaLore")]:
            if key in self.results:
                techniques.append(name)
                bleu_scores.append(self.results[key].get("bleu_4", 0))
                rouge_scores.append(self.results[key].get("rouge_l", 0))

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # BLEU-4
        ax1.bar(techniques, bleu_scores, color=['gray', 'blue', 'green'][:len(techniques)])
        ax1.set_ylabel('BLEU-4 Score')
        ax1.set_title('BLEU-4 Comparison')
        ax1.set_ylim([0, max(bleu_scores) * 1.2 if bleu_scores else 1])
        for i, v in enumerate(bleu_scores):
            ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

        # ROUGE-L
        ax2.bar(techniques, rouge_scores, color=['gray', 'blue', 'green'][:len(techniques)])
        ax2.set_ylabel('ROUGE-L Score')
        ax2.set_title('ROUGE-L Comparison')
        ax2.set_ylim([0, max(rouge_scores) * 1.2 if rouge_scores else 1])
        for i, v in enumerate(rouge_scores):
            ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plot_file = os.path.join(output_dir, "bleu_rouge_comparison.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"BLEU/ROUGE graph saved: {plot_file}")
        plt.close()

    def plot_memory_vs_performance(self, output_dir: str = "./results/plots"):
        """Memory vs Performance graph (assessment requirement)"""
        os.makedirs(output_dir, exist_ok=True)

        techniques = []
        memory_usage = []
        bleu_scores = []
        rouge_scores = []

        for key, name in [("qlora", "QLoRA"), ("galore", "GaLore")]:
            if key in self.results:
                result = self.results[key]
                if result.get("peak_memory_gb"):
                    techniques.append(name)
                    memory_usage.append(result.get("peak_memory_gb", 0))
                    bleu_scores.append(result.get("bleu_4", 0))
                    rouge_scores.append(result.get("rouge_l", 0))

        if not techniques:
            print("Memory data not found")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # BLEU vs Memory
        ax1.scatter(memory_usage, bleu_scores, s=200, alpha=0.6)
        for i, txt in enumerate(techniques):
            ax1.annotate(txt, (memory_usage[i], bleu_scores[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Peak Memory (GB)')
        ax1.set_ylabel('BLEU-4 Score')
        ax1.set_title('BLEU-4 vs Memory Usage')
        ax1.grid(True, alpha=0.3)

        # ROUGE vs Memory
        ax2.scatter(memory_usage, rouge_scores, s=200, alpha=0.6, color='green')
        for i, txt in enumerate(techniques):
            ax2.annotate(txt, (memory_usage[i], rouge_scores[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Peak Memory (GB)')
        ax2.set_ylabel('ROUGE-L Score')
        ax2.set_title('ROUGE-L vs Memory Usage')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, "memory_vs_performance.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Memory vs Performance graph saved: {plot_file}")
        plt.close()

    def create_summary_report(self, output_dir: str = "./results"):
        """Creates summary report"""
        os.makedirs(output_dir, exist_ok=True)

        report = []
        report.append("="*80)
        report.append("FINE-TUNING COMPARISON SUMMARY REPORT")
        report.append("="*80)
        report.append("")

        if "base" in self.results:
            report.append("BASE MODEL PERFORMANCE:")
            report.append(f"  BLEU-4:  {self.results['base'].get('bleu_4', 0):.4f}")
            report.append(f"  ROUGE-L: {self.results['base'].get('rouge_l', 0):.4f}")
            report.append("")

        for key, name in [("qlora", "QLoRA"), ("galore", "GaLore")]:
            if key in self.results:
                result = self.results[key]
                report.append(f"{name.upper()} PERFORMANCE:")
                report.append(f"  BLEU-4:  {result.get('bleu_4', 0):.4f}")
                report.append(f"  ROUGE-L: {result.get('rouge_l', 0):.4f}")

                if "base" in self.results:
                    bleu_improvement = result.get('bleu_4', 0) - self.results['base'].get('bleu_4', 0)
                    rouge_improvement = result.get('rouge_l', 0) - self.results['base'].get('rouge_l', 0)
                    report.append(f"  BLEU-4 Improvement:  {bleu_improvement:+.4f}")
                    report.append(f"  ROUGE-L Improvement: {rouge_improvement:+.4f}")

                if result.get('peak_memory_gb'):
                    report.append(f"  Peak Memory: {result.get('peak_memory_gb'):.2f} GB")
                if result.get('training_time_hours'):
                    report.append(f"  Training Time: {result.get('training_time_hours'):.2f} hours")
                report.append("")

        report.append("="*80)

        report_text = "\n".join(report)
        print("\n" + report_text)

        report_file = os.path.join(output_dir, "summary_report.txt")
        with open(report_file, "w") as f:
            f.write(report_text)

        print(f"\nSummary report saved: {report_file}")


def main():
    """Main function"""
    print("="*60)
    print("Results Visualization")
    print("="*60)

    visualizer = ResultsVisualizer()
    visualizer.load_results()

    visualizer.create_comparison_table()
    visualizer.plot_bleu_rouge_comparison()
    visualizer.plot_memory_vs_performance()
    visualizer.create_summary_report()

    print("\n" + "="*60)
    print("Visualization Completed")
    print("="*60)


if __name__ == "__main__":
    main()
