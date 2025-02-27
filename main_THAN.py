import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, ndcg_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils_THAN import load_data, EarlyStopping, get_binary_mask

# Regularización L2 --> #Presta atención a configurar el coeficiente de la regularización: normalmente se elige 0.001

def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, param in model.named_parameters():
        if 'bias' not in name:  # 一般不对偏置项使用正则  Normalmente no se aplica regularización al término de sesgo.
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(torch.pow(param, 2)))
    return l2_loss

# Clasificación multiclase -> f1_weight
def get_weighted_fscore(dic_, y_pred, y_true):
    f_score = 0
    for i in range(11):
        yt = y_true == i
        yp = y_pred == i
        f_score += dic_[i] * f1_score(y_true=yt, y_pred=yp)
    return f_score

def score(logits, labels):
    # Se cuentan las proporciones que ocupa cada valor en las etiquetas
    df_analysis = pd.DataFrame()
    df_analysis['label'] = labels
    dic_ = df_analysis['label'].value_counts(normalize=True)
    # logits es un arreglo bidimensional de [tamaño del conjunto de entrenamiento] x [número de modos de transporte]; cada fila
    # “_” corresponde a la probabilidad o puntuación de cada modo de transporte, e “indices” corresponde al id con la puntuación más alta (modo de transporte)
    _, indices = torch.max(logits, dim=1)
    # Etiquetas pronosticadas
    prediction = indices.long().cpu().numpy()
    # Etiquetas reales del nodo
    labels = labels.cpu().numpy()

    # Para cada etiqueta, la puntuación corresponde a un arreglo bidimensional
    y_score = np.zeros([1, logits.shape[0]], dtype=np.float64)
    for i in range(0, logits.shape[0]):
        y_score[0][i] = logits[i][prediction[i]]

    # Se reasignan las etiquetas para formar un arreglo bidimensional
    y_values = np.zeros([1, labels.shape[0]], dtype=float)
    for i in range(0, labels.shape[0]):
        y_values[0][i] = labels[i]
    # Obtener la métrica F1 para clasificación multiclase
    f_score = get_weighted_fscore(dic_, labels, prediction)

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    f1_weighted = f1_score(labels, prediction, average='weighted')
    Pre = precision_score(labels, prediction, average='weighted', zero_division=1)
    Rec = recall_score(labels, prediction, average='weighted')
    # Los valores de entrada son arreglos bidimensionales; la función llamada ya se ha explicado, ¡presta atención!
    NDCG = ndcg_score(y_values, y_score)

    return accuracy, micro_f1, macro_f1, f1_weighted, Pre, Rec, NDCG, f_score

def evaluate(model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, mask, loss_func, o_d_od_ID_data, o_d_count):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, pid_features, o_features, d_features, o_d_g, d_o_g, od_features, o_d_od_ID_data, o_d_count)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1, f1_weighted, Pre, Rec, NDCG, f_score = score(logits[mask], labels[mask])
    _, indices = torch.max(logits[mask], dim=1)
    # Etiquetas pronosticadas
    prediction = indices.long().cpu().numpy()
    # Etiquetas reales del nodo
    labels_new = labels[mask]
    labels_new = labels_new.cpu().numpy()

    return loss, accuracy, micro_f1, macro_f1, f1_weighted, Pre, Rec, NDCG, f_score, labels_new, prediction

def main(args):
    g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, test_mask_0, test_mask_1, test_mask_2, test_mask_3, o_d_count = load_data("Mydataset")

    # Parece crucial; es necesario procesar el valor de mask aquí, ¿no?
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        test_mask_0 = test_mask_0.bool()
        test_mask_1 = test_mask_1.bool()
        test_mask_2 = test_mask_2.bool()
        test_mask_3 = test_mask_3.bool()

    # Características de mode
    features = features.to(args['device'])
    # Características de pid
    pid_features = pid_features.to(args['device'])
    # Características de od
    o_features = o_features.to(args['device'])
    d_features = d_features.to(args['device'])
    od_features = od_features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])

    test_mask = test_mask.to(args['device'])
    test_mask_0 = test_mask_0.to(args['device'])
    test_mask_1 = test_mask_1.to(args['device'])
    test_mask_2 = test_mask_2.to(args['device'])
    test_mask_3 = test_mask_3.to(args['device'])
    o_d_count = o_d_count.to(args['device'])

    from model_THAN import THAN
    # Incluye pid y od
    model = THAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                in_size=features.shape[1],
                # Dimensión de características pid
                pid_size=pid_features.shape[1],
                # Dimensión de características od
                o_size=o_features.shape[1],
                d_size=d_features.shape[1],
                od_size=od_features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    g = g.to(args['device'])
    dropout = args['dropout']
    # Grafo bipartito entre o y d
    o_d_g = o_d_g.to(args['device'])
    d_o_g = d_o_g.to(args['device'])
    # Definiciones relacionadas: early stopping, función de pérdida, optimizador
    stopper = EarlyStopping(patience=args['patience'])
    # Función de pérdida
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])
    # your computer's path
    path = '../p38dglproject/dataset/output/'
    who = 'beijing'
    o_d_od_ID_data = pd.read_csv((path + who + '/o_d_od_ID_data.csv'))
    o_d_od_ID_data = o_d_od_ID_data.values
    o_d_od_ID_data = torch.FloatTensor(o_d_od_ID_data)
    o_d_od_ID_data = o_d_od_ID_data.to(args['device'])
    #  Iteración de optimización
    Train_dataloader = DataLoader(train_mask, batch_size=20, drop_last=False)
    for epoch in range(args['num_epochs']):
        # 批处理
        # for step, batch_train_feature, batch_train_mask in enumerate(Train_dataloader):
        model.train()
        # -->数据量太大的时候，可运用批处理技术解决
        logits = model(g, features, pid_features, o_features, d_features, o_d_g, d_o_g, od_features, o_d_od_ID_data, o_d_count)
        # -->logits是每项sid对应每种交通方式的概率值[mode0~mode10]
        # -->损失函数
        CrossEntropy = loss_fcn(logits[train_mask], labels[train_mask])
        # L2Loss为正则项，主要为了放着过拟合
        loss = CrossEntropy # + L2Loss(model, 0.001)
        # 反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1, train_f1_weighted, train_Pre, train_Rec, train_NDCG, train_f_score = score(
            logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1, val_f1_weighted, val_Pre, val_Rec, val_NDCG, val_f_score, _, _ = evaluate(model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, val_mask, loss_fcn, o_d_od_ID_data, o_d_count)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print(
        'Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | Train f1_weighted {:.4f} | Train Pre {:.4f} | Train Rec {:.4f} | Train NDCG {:.4f} | Train f1_score {:.4f} |'
        'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val f1_weighted {:.4f} | Val Pre {:.4f} | Val Rec {:.4f} | Val NDCG {:.4f} | Val f1_score {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, train_f1_weighted, train_Pre, train_Rec,
            train_NDCG, train_f_score, val_loss.item(), val_micro_f1, val_macro_f1, val_f1_weighted, val_Pre,
            val_Rec, val_NDCG, val_f_score))
        if early_stop:
            break
    # -->导入训练好的模型
    stopper.load_checkpoint(model)

    test_loss, test_acc, test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG, test_f_score,labels_new, prediction = evaluate(
        model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, test_mask,
        loss_fcn, o_d_od_ID_data, o_d_count)
    print(
        'Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test f1_weighted {:.4f} | Test Pre {:.4f} | Test Rec {:.4f} | Test NDCG {:.4f} | Test f1_score {:.4f}'.format(
            test_loss.item(), test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG,
            test_f_score))

    test_loss, test_acc, test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG, test_f_score, _, _ = evaluate(
        model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, test_mask_0, loss_fcn, o_d_od_ID_data, o_d_count)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test f1_weighted {:.4f} | Test Pre {:.4f} | Test Rec {:.4f} | Test NDCG {:.4f} | Test f1_score {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG, test_f_score))

    test_loss, test_acc, test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG, test_f_score, _, _ = evaluate(
        model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, test_mask_1,
        loss_fcn, o_d_od_ID_data, o_d_count)
    print(
        'Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test f1_weighted {:.4f} | Test Pre {:.4f} | Test Rec {:.4f} | Test NDCG {:.4f} | Test f1_score {:.4f}'.format(
            test_loss.item(), test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG,
            test_f_score))
    test_loss, test_acc, test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG, test_f_score, _, _ = evaluate(
        model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, test_mask_2,
        loss_fcn, o_d_od_ID_data, o_d_count)
    print(
        'Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test f1_weighted {:.4f} | Test Pre {:.4f} | Test Rec {:.4f} | Test NDCG {:.4f} | Test f1_score {:.4f}'.format(
            test_loss.item(), test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG,
            test_f_score))
    test_loss, test_acc, test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG, test_f_score, _, _ = evaluate(
        model, g, o_d_g, d_o_g, features, pid_features, o_features, d_features, od_features, labels, test_mask_3,
        loss_fcn, o_d_od_ID_data, o_d_count)
    print(
        'Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test f1_weighted {:.4f} | Test Pre {:.4f} | Test Rec {:.4f} | Test NDCG {:.4f} | Test f1_score {:.4f}'.format(
            test_loss.item(), test_micro_f1, test_macro_f1, test_f1_weighted, test_Pre, test_Rec, test_NDCG,
            test_f_score))

    confusion_mat_new = confusion_matrix(labels_new, prediction)
    confusion_mat = confusion_mat_new.astype('float') / confusion_mat_new.sum(axis=1)[:, np.newaxis]
    confusion_mat = np.around(confusion_mat, decimals=2)
    classes = ["1", "2", "3", "4", "5", "6", "7", "8",
               "9", "10", "11"]
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=classes)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation="horizontal", values_format=".2g")
    plt.title("Normalized confusion matrix of THAN")
    plt.show()

if __name__ == '__main__':
    import argparse
    from utils_THAN import setup
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
