
import numpy as np
import cv2
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelUtils:
    """Utilidades para el modelo"""
    
    @staticmethod
    def load_model_config(config_path='results/model_results.json'):
        """Carga configuración del modelo"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_normalization_stats(stats_path='processed_dataset/normalization_stats.json'):
        """Carga estadísticas de normalización"""
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return {
            'mean': np.array(stats['mean_rgb']) / 255.0,
            'std': np.array(stats['std_rgb']) / 255.0,
            'target_size': tuple(stats['target_size']),
            'label_map': stats['label_map']
        }
    
    @staticmethod
    def preprocess_for_prediction(image_path, mean, std, target_size=(224, 224)):
        """Preprocesa una imagen para predicción"""
        # Cargar imagen
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, target_size)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        
        # Añadir dimensión batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    @staticmethod
    def visualize_prediction(image_path, prediction, class_names, save_path=None):
        """Visualiza una predicción"""
        # Cargar imagen original
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Mostrar imagen
        axes[0].imshow(img)
        axes[0].axis('off')
        predicted_class = class_names[np.argmax(prediction)]
        confidence = prediction[np.argmax(prediction)]
        axes[0].set_title(f'Predicción: {predicted_class}\nConfianza: {confidence:.2%}',
                         fontsize=14, fontweight='bold')
        
        # Mostrar probabilidades
        axes[1].barh(class_names, prediction, color=['#FF6B6B', '#4ECDC4'])
        axes[1].set_xlabel('Probabilidad', fontweight='bold')
        axes[1].set_title('Distribución de Probabilidades', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Añadir valores
        for i, prob in enumerate(prediction):
            axes[1].text(prob, i, f'{prob:.2%}', 
                        ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class DatasetUtils:
    """Utilidades para el dataset"""
    
    @staticmethod
    def get_dataset_statistics(df):
        """Calcula estadísticas del dataset"""
        stats = {
            'total_images': len(df),
            'class_distribution': df['labels'].value_counts().to_dict(),
            'avg_width': df['width'].mean(),
            'avg_height': df['height'].mean(),
            'avg_aspect_ratio': df['aspect_ratio'].mean()
        }
        return stats
    
    @staticmethod
    def plot_class_distribution(df, save_path=None):
        """Grafica distribución de clases"""
        plt.figure(figsize=(10, 6))
        
        counts = df['labels'].value_counts()
        plt.bar(counts.index, counts.values, color=['#FF6B6B', '#4ECDC4'], 
                alpha=0.7, edgecolor='black')
        
        plt.xlabel('Clase', fontweight='bold', fontsize=12)
        plt.ylabel('Número de Imágenes', fontweight='bold', fontsize=12)
        plt.title('Distribución de Clases en el Dataset', 
                 fontweight='bold', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        # Añadir valores
        for i, (idx, val) in enumerate(counts.items()):
            plt.text(i, val + 1, str(val), ha='center', 
                    va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_balanced_dataset(df, target_samples_per_class=None):
        """Crea un dataset balanceado"""
        if target_samples_per_class is None:
            # Usar el mínimo de todas las clases
            target_samples_per_class = df['labels'].value_counts().min()
        
        balanced_dfs = []
        for label in df['labels'].unique():
            class_df = df[df['labels'] == label].sample(
                n=min(target_samples_per_class, len(df[df['labels'] == label])),
                random_state=42
            )
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(
            frac=1, random_state=42
        ).reset_index(drop=True)

class EvaluationUtils:
    """Utilidades para evaluación"""
    
    @staticmethod
    def plot_confusion_matrix_detailed(cm, class_names, save_path=None):
        """Grafica matriz de confusión detallada"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Matriz absoluta
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0], cbar_kws={'label': 'Número de muestras'},
                   linewidths=2, linecolor='black')
        axes[0].set_xlabel('Predicción', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Valor Real', fontweight='bold', fontsize=12)
        axes[0].set_title('Matriz de Confusión (Valores Absolutos)',
                         fontweight='bold', fontsize=14)
        
        # Matriz normalizada
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1], cbar_kws={'label': 'Porcentaje'},
                   linewidths=2, linecolor='black', vmin=0, vmax=1)
        axes[1].set_xlabel('Predicción', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Valor Real', fontweight='bold', fontsize=12)
        axes[1].set_title('Matriz de Confusión (Normalizada)',
                         fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def calculate_per_class_metrics(cm, class_names):
        """Calcula métricas por clase"""
        metrics = {}
        
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            accuracy = (tp + tn) / cm.sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }
        
        return metrics
    
    @staticmethod
    def print_metrics_report(metrics):
        """Imprime reporte de métricas"""
        print("\n" + "="*70)
        print("REPORTE DE MÉTRICAS POR CLASE")
        print("="*70)
        
        for class_name, class_metrics in metrics.items():
            print(f"\n📊 Clase: {class_name}")
            print(f"  Accuracy:  {class_metrics['accuracy']:.4f} ({class_metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1-Score:  {class_metrics['f1_score']:.4f}")
            print(f"  TP: {class_metrics['true_positives']}, "
                  f"FP: {class_metrics['false_positives']}, "
                  f"FN: {class_metrics['false_negatives']}, "
                  f"TN: {class_metrics['true_negatives']}")