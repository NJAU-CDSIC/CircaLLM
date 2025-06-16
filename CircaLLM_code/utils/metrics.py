import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import json
import numpy
import numpy as np

class Metric:
    #用于二分类问题
    @staticmethod
    def plot_roc_curve(labels, scores):#ROC
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        roc_auc = metrics.auc(fpr, tpr)#ROC曲线的值，AUC
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    #用于二分类或多分类问题
    @staticmethod
    def plot_pr_curve(labels:numpy.ndarray, scores:numpy.ndarray):#PR
        precision, recall, _ = metrics.precision_recall_curve(labels, scores)
        ap = metrics.average_precision_score(labels, scores)
        plt.plot(recall, precision, label=f'Class 1 (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for Single Class Classification')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def plot_2class_pr_curve(labels:numpy.ndarray, scores:numpy.ndarray):
        labels_class1 = (labels == 1).astype(int)
        labels_class2 = (labels == 0).astype(int)

        precision1, recall1, _ = metrics.precision_recall_curve(labels_class1, scores[:,1])
        precision2, recall2, _ = metrics.precision_recall_curve(labels_class2, scores[:,0])

        ap1 = metrics.average_precision_score(labels_class1, scores[:,1])
        ap2 = metrics.average_precision_score(labels_class2, scores[:,0])
        plt.plot(recall1, precision1, label=f'Class per (AP = {ap1:.4f})')
        plt.plot(recall2, precision2, label=f'Class aper (AP = {ap2:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for Two Classes Classification')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    @staticmethod
    def calculate_map(labels, scores):
        map_score = metrics.average_precision_score(labels, scores, average='macro')  # 'macro' for multi-class
        print(f"mAP: {map_score:.4f}")
    #用于二分类或多分类问题
    @staticmethod
    def get_classification_report(labels, predictions):#可以对于每个分类的类别，给出F1-score,precision,recall
        report = metrics.classification_report(labels, predictions,digits=4)
        return report
        # print(report)
    @staticmethod
    def save_metrics(result, addr, filename,epoch,c_time,mode='a'):
        result['epoch']=epoch
        saveAddr=os.path.join(addr,c_time,filename)
        
        # 获取文件所在的目录
        directory = os.path.dirname(saveAddr)
        # 检查目录是否存在，如果不存在，则创建它
        if not os.path.exists(directory):
            os.makedirs(directory)
        if mode.lower()=="a":
            if not os.path.exists(saveAddr) or os.path.getsize(saveAddr) == 0:
                # 文件不存在或为空，写入第一个元素，并格式化为JSON数组
                with open(saveAddr, 'w') as f:
                    json.dump([result], f, indent=4)
            else:
                del result['targets']
                with open(saveAddr, 'r+') as file:
                    file.seek(0, 2)  # 移动到文件末尾
                    position = file.tell()
                    while position >= 0:
                        file.seek(position)
                        if file.read(1) == '\n':  # 检查是否是换行符
                            break
                        position -= 1
                    file.seek(position)
                    file.write(',\n' +json.dumps([result], indent=4)[2:])
        elif mode.lower()=='w':
            with open(saveAddr, 'w') as f:
                    json.dump(result, f, indent=4)
        print(f"Epoch {epoch} results saved to {saveAddr}")
    @staticmethod
    def plot_loss(train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    @staticmethod
    def plot_accuracy(train_accuracies, val_accuracies):
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()