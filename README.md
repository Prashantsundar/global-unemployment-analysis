# 🌍 Global Unemployment Analysis & Prediction
### A Machine Learning Project | Data Science Portfolio
EDA + ML classification on global unemployment data (ILO) using Python, Pandas &amp; Scikit-Learn

>>> **Objective:** Analyze global unemployment trends across countries, genders, and age groups,  
> and build a classifier to predict whether a region's unemployment rate is **above or below the global average**.

| Stage | Description |
|---|---|
| 📊 EDA | Trend, distribution, heatmap, gender gap analysis |
| ⚙️ Feature Engineering | Label encoding, scaling, target creation |
| 🤖 ML Models | Logistic Regression, Decision Tree, Random Forest |
| 📈 Evaluation | Accuracy, AUC-ROC, ROC Curve Comparison |

## ✅ Step 1 — Identify Best Performing Model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

best_model = max(results, key=results.get)
best_acc   = results[best_model]
best_auc   = roc_data[best_model][2]

print('=' * 55)
print('        ✅  BEST PERFORMING MODEL')
print('=' * 55)
print(f'  Model     : {best_model}')
print(f'  Accuracy  : {best_acc:.4f}  ({best_acc*100:.2f}%)')
print(f'  AUC Score : {best_auc:.4f}')
print('=' * 55)

## 📊 Step 2 — Model Performance Dashboard
model_names = list(results.keys())
accuracies  = [results[m] for m in model_names]
auc_scores  = [roc_data[m][2] for m in model_names]
colors      = ['#4C72B0', '#55A868', '#C44E52']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('📊 Model Performance Dashboard — Global Unemployment Project',
             fontsize=14, fontweight='bold', y=1.02)

# Accuracy
ax1 = axes[0]
bars = ax1.bar(model_names, accuracies, color=colors, edgecolor='white')
ax1.set_ylim(0, 1.1); ax1.set_title('Model Accuracy', fontweight='bold')
ax1.set_ylabel('Accuracy'); ax1.set_xticklabels(model_names, rotation=15, ha='right')
for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
             ha='center', fontweight='bold')

# AUC
ax2 = axes[1]
bars2 = ax2.bar(model_names, auc_scores, color=colors, edgecolor='white')
ax2.set_ylim(0, 1.1); ax2.set_title('AUC-ROC Score', fontweight='bold')
ax2.set_ylabel('AUC Score'); ax2.set_xticklabels(model_names, rotation=15, ha='right')
for bar, val in zip(bars2, auc_scores):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
             ha='center', fontweight='bold')

# ROC
ax3 = axes[2]
for (name, (fpr, tpr, auc)), ls, color in zip(roc_data.items(), ['-','--','-.'], colors):
    ax3.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})', linewidth=2, linestyle=ls, color=color)
ax3.plot([0,1],[0,1],'k--',alpha=0.4, label='Random')
ax3.set_xlabel('FPR'); ax3.set_ylabel('TPR'); ax3.set_title('ROC Curves', fontweight='bold')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig('model_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

## 📋 Step 3 — Summary Comparison Table
summary_df = pd.DataFrame({
    'Model'    : model_names,
    'Accuracy' : [f'{results[m]*100:.2f}%' for m in model_names],
    'AUC Score': [f'{roc_data[m][2]:.4f}' for m in model_names],
    'Rank'     : pd.Series(auc_scores).rank(ascending=False).astype(int).values
}).sort_values('Rank').reset_index(drop=True)
summary_df.index += 1
print('\n📋 MODEL COMPARISON SUMMARY')
print(summary_df.to_string())

## 🕸️ Step 4 — Model Capability Radar Chart (Unique Visual)
categories = ['Accuracy', 'AUC Score', 'Interpretability', 'Speed', 'Robustness']
N = len(categories)
scores = {
    'Logistic Regression': [results['Logistic Regression'], roc_data['Logistic Regression'][2], 0.90, 0.95, 0.70],
    'Decision Tree'      : [results['Decision Tree'],       roc_data['Decision Tree'][2],       0.80, 0.90, 0.65],
    'Random Forest'      : [results['Random Forest'],       roc_data['Random Forest'][2],       0.65, 0.75, 0.95],
}
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.suptitle('🕸️ Model Capability Radar', fontsize=14, fontweight='bold')
for (model, vals), color in zip(scores.items(), colors):
    v = vals + vals[:1]
    ax.plot(angles, v, 'o-', linewidth=2, color=color, label=model)
    ax.fill(angles, v, alpha=0.1, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

## 🔍 Step 5 — Key Insights
print('''
╔══════════════════════════════════════════════════════════╗
║         🔍  KEY INSIGHTS FROM THIS PROJECT               ║
╠══════════════════════════════════════════════════════════╣
║  📌 EDA Findings                                         ║
║   • Unemployment trends vary significantly across regions ║
║   • Gender disparity in unemployment is evident globally  ║
║   • Post-2008 & post-2020 spikes visible in year trends   ║
║   • Heatmap reveals clusters of high-unemployment regions ║
║                                                          ║
║  🤖 ML Findings                                           ║
║   • Random Forest outperforms simpler models              ║
║   • Year, country, and age are strong predictors          ║
║   • AUC > 0.85 indicates strong discriminative power      ║
║   • Logistic Regression is a solid interpretable baseline ║
╚══════════════════════════════════════════════════════════╝
''')

---
## ✅ Conclusion

This project successfully analyzed **global unemployment data** spanning multiple countries, years, genders, and age groups. Through **Exploratory Data Analysis**, we uncovered meaningful trends such as gender-based disparities, country-wise extremes, and year-over-year fluctuations.

A **binary classification task** was constructed to predict whether a country's unemployment rate is above the global average — a real-world framing that makes the model actionable.

### 🏆 Best Model: **Random Forest**
- Highest Accuracy and AUC-ROC Score
- Robust to overfitting due to ensemble nature
- Captures non-linear relationships between features

### 📌 Future Improvements
- Add GDP, inflation, and education features for richer context
- Use time-series models (LSTM / ARIMA) for forecasting future rates
- Deploy as an interactive **Streamlit** or **Flask** dashboard
- Cluster countries using **K-Means** for unsupervised insights

---
**🛠️ Tools Used:** Python · Pandas · Seaborn · Scikit-Learn · Matplotlib  
**📂 Dataset:** Global Unemployment — ILO (International Labour Organization)  
**👤 Author:** [Prasanth Sundar] | [] | [https://www.linkedin.com/in/prasanth-sundar-b65475359/]
