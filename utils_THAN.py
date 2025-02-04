import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import pandas as pd
from sklearn import preprocessing

from NodeFeature import *

from sklearn.preprocessing import OneHotEncoder

# Transformación lineal -> reducción de dimensiones
# Entrada: el tensor que se desea reducir y la dimensión de salida
# Salida: retorna el tensor con la dimensión modificada
def LinearChange(res, dst_feat_D):
    # print("res =", res, res.shape, dst_feat_D)
    fc = nn.Linear(res.shape[1], dst_feat_D, bias=True)
    Ba = nn.BatchNorm1d(dst_feat_D)
    Dr = nn.Dropout(0.2)

    dst_feat = fc(res)
    # print("dst_feat =", dst_feat, dst_feat.shape)
    dst_feat = Ba(dst_feat)
    dst_feat = F.leaky_relu(dst_feat)
    # dst_feat = F.softmax(dst_feat, dim=1)
    dst_feat = Dr(dst_feat)
    return dst_feat

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,  # Learning rate
    'num_heads': [8],  # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,  # Regularización https://zhuanlan.zhihu.com/p/62393636
    'num_epochs': 200,
    'patience': 100
}

sampling_configure = {
    'batch_size': 20
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    #args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['device'] = 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    #args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['device'] = 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

# python main.py 
def load_mydata_raw(remove_self_loop):  ####################################
    assert not remove_self_loop
    data_path = get_download_dir() + '/mydataset.mat'
    print("mydataset.mat put in:", data_path)
    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field? ¿campo de artículos? área de investigación del artículo
    p_vs_a = data['PvsA']  # paper-author autor(es) del artículo
    p_vs_t = data['PvsT']  # paper-term, bag of words términos del artículo -> palabras relevantes en el texto
    p_vs_c = data['PvsC']  # paper-conference, labels come from that conferencia del artículo, de ahí provienen las etiquetas
    num = p_vs_l.shape[0]
    # test data number
    # num = 308507
    # num = 50000
    # num = 10000
    # num = 5000
    p_vs_l = p_vs_l[:num]
    p_vs_a = p_vs_a[:num]
    p_vs_t = p_vs_t[:num]
    p_vs_c = p_vs_c[:num]
    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    # conf_ids = [0, 1, 9, 10, 13]
    conf_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # label_ids = [0, 1, 2, 2, 1]
    label_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    # od
    p_vs_l = p_vs_l[p_selected]
    # usuario
    p_vs_a = p_vs_a[p_selected]
    # características o atributos del elemento correspondiente (session id, SID)
    p_vs_t = p_vs_t[p_selected]  # -->Esto cambiará más adelante; la idea previa de usar la etiqueta como atributo no era correcta
    # print("len(p_vs_t) =", p_vs_t.shape)

    # Clasificación a la que pertenece
    p_vs_c = p_vs_c[p_selected]
    # Construir el grafo
    hg = dgl.heterograph({
        # modo de transporte -> usuario?
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        # modo de transporte -> par od?
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })
    # cantidad de usuarios
    pid_num = hg.number_of_nodes('author')
    # cantidad de od
    od_num = hg.number_of_nodes('field')
    x = hg.edges('all', etype='pa')
    y = hg.edges('all', etype='pf')
    # Atributo espacial
    od_distance = pd.read_csv("../p38dglproject/dataset/output/beijing/train_click_od.csv", usecols=['manhattan'])
    # Atributo temporal
    timedata = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv",
                           usecols=['req_time_hour', 'req_time_weekday', 'elapsed_time', 'minute', 'month', 'day'])
    # Atributos meteorológicos
    max_min_temp = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv",
                               usecols=['max_temp', 'min_temp', 'wind'])
    weatherdata = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv",
                              usecols=['weather'])
    winddata = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv",
                           usecols=['wind'])
    # Atributos de plan
    # -->Se ha demostrado experimentalmente que plan_mode_fea es muy importante
    plan_mode_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv",
                                usecols=['mode_feas_1', 'mode_feas_2', 'mode_feas_3', 'mode_feas_4', 'mode_feas_5',
                                         'mode_feas_6', 'mode_feas_7', 'mode_feas_8', 'mode_feas_9', 'mode_feas_10',
                                         'mode_feas_11'])
    plan_firstmode_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['first_mode'])
    plan_speed_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv",
                                 usecols=['speed_feas_1', 'speed_feas_2', 'speed_feas_3', 'speed_feas_4',
                                          'speed_feas_5', 'speed_feas_6', 'speed_feas_7', 'speed_feas_8',
                                          'speed_feas_9', 'speed_feas_10', 'speed_feas_11'])

    plan_price_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv",
                                 usecols=['price_feas_1', 'price_feas_2', 'price_feas_3', 'price_feas_4',
                                          'price_feas_5', 'price_feas_6', 'price_feas_7', 'price_feas_8',
                                          'price_feas_9', 'price_feas_10', 'price_feas_11'])

    plan_energy_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv",
                                  usecols=['energy_feas_1', 'energy_feas_2', 'energy_feas_3', 'energy_feas_4',
                                           'energy_feas_5', 'energy_feas_6', 'energy_feas_7', 'energy_feas_8',
                                           'energy_feas_9', 'energy_feas_10', 'energy_feas_11'])

    # Características espaciales
    od_distance_feature = od_distance.values.astype(np.int)
    
    # Características temporales
    timefeature = timedata.values

    # Características meteorológicas
    max_min_temparrany = max_min_temp.values
    weatherdataarrany = weatherdata.values
    winddataarrany = winddata.values

    # Características del plan
    plan_mode_feaarrany = plan_mode_fea.values

    # Se hace onehot a la modalidad “first_mode”, intentando restar 1; en .mat se restan 1 a los modos
    plan_firstmode_feaarrany = plan_firstmode_fea.values - 1
    # Codificación de velocidad
    plan_speed_feaarrany = plan_speed_fea.values.astype(np.int)
    plan_price_feaarrany = (plan_price_fea.values / 100).astype(np.int)
    # Tipos de energía usados por el modo de transporte
    plan_energy_feaarrany = plan_energy_fea.values
    # Estandarización -> puede ser útil para clasificación
    scaler = preprocessing.StandardScaler()
    # Se podría usar la normalización MinMax para mapear todos los valores a [0,1]
    min_max_scaler = preprocessing.MinMaxScaler()
    # ------>Antes se usó MinMaxScaler aquí<----------
    # Parece que normalizar los valores de velocidad da mejor resultado que onehot
    plan_speed_feaarrany = min_max_scaler.fit_transform(plan_speed_feaarrany)
    # Con MinMaxScaler se logra un resultado de 0.7422
    max_min_temparrany = min_max_scaler.fit_transform(max_min_temparrany)
    winddataarrany = min_max_scaler.fit_transform(winddataarrany)

    # ---->Usar onehot en general mejora los resultados, ya que la distancia euclidiana funciona mejor; aunque ocupa más espacio
    # ------>Si hay suficiente memoria, se podría usar onehot para todo<------#
    # Se realiza onehot encoding para las características de od
    enc = OneHotEncoder(sparse=False)
    # tiempo
    timefeature = enc.fit_transform(timefeature)

    # El valor de velocidad da mejor resultado con normalización (ya en MinMax)
    plan_firstmode_feaarrany = enc.fit_transform(plan_firstmode_feaarrany)
    plan_price_feaarrany = enc.fit_transform(plan_price_feaarrany)
    plan_energy_feaarrany = enc.fit_transform(plan_energy_feaarrany)
    od_distance_feature = enc.fit_transform(od_distance_feature)

    # meteorología
    # Para el clima, que son strings, seguimos usando onehot
    weatherdataarrany = enc.fit_transform(weatherdataarrany)
    # Temperatura -> parece que el escalado MinMax es mejor
    max_min_temparrany = enc.fit_transform(max_min_temparrany)
    # Fuerza del viento -> con datos originales va un poco mejor, pero también sirve MinMax
    winddataarrany = enc.fit_transform(winddataarrany)

    # Características de asociación entre modos de transporte

    # -------------------->Características importantes<------------------------#

    # -->Atributos de usuario, tiempo, espacio (ubicación, par od), y atributos en plan: modo, velocidad, primer modo, precio    
    # pid_feat_feature
    # od_feat_feature
    # Atributos temporales
    termfeature = timefeature

    # Atributos en plan: modo, velocidad, primer modo, precio
    termfeature = np.concatenate((termfeature, plan_mode_feaarrany), axis=1)
    termfeature = np.concatenate((termfeature, plan_speed_feaarrany), axis=1)
    termfeature = np.concatenate((termfeature, plan_firstmode_feaarrany), axis=1)
    termfeature = np.concatenate((termfeature, plan_price_feaarrany), axis=1)

    # Distintos modos de transporte usan distintos tipos de energía
    termfeature = np.concatenate((termfeature, plan_energy_feaarrany), axis=1)
    # -->La distancia influye en la elección del modo de transporte; p. ej., menos de 10km se prefiere caminar o bicicleta
    termfeature = np.concatenate((termfeature, od_distance_feature), axis=1)
    # Atributos meteorológicos -> al incorporarlos todos, mejora el rendimiento; si se ponen aislados, parece empeorar
    
    # -->Estado del clima
    termfeature = np.concatenate((termfeature, weatherdataarrany), axis=1)
    # -->Temperatura máxima y mínima
    termfeature = np.concatenate((termfeature, max_min_temparrany), axis=1)
    # -->Intensidad del viento -> 0.7058
    termfeature = np.concatenate((termfeature, winddataarrany), axis=1)

    # --------->Guardar las características originales generadas actualmente<--------#
    # Atributos originales -> principalmente atributos de usuario p0-p65
    origin_feature = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_nonoise.csv")
    # Este atributo no incluye sid, req_time, o, d, plan_time, plans, click_time, click_mode, city_flag_o
    del origin_feature['sid']
    # del origin_feature['pid']
    del origin_feature['req_time']
    del origin_feature['o']
    del origin_feature['d']
    del origin_feature['plan_time']
    del origin_feature['plans']
    del origin_feature['click_time']
    # Eliminar filas duplicadas según pid
    del origin_feature['o_lng']
    del origin_feature['o_lat']
    del origin_feature['d_lng']
    del origin_feature['d_lat']
    del origin_feature['click_mode']
    del origin_feature['city_flag_o']

    # Eliminar filas duplicadas según pid
    origin_feature = origin_feature.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    del origin_feature['pid']

    # Ya está en onehot
    origin_feature_new = origin_feature.values
    origin_feature_new = origin_feature_new[:pid_num]

    print("origin_feature_new =", origin_feature_new, origin_feature_new.shape)

    # Atributos espaciales (o y d) -> características de d, o, pid, time

    o_d_lng_lat = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_nonoise.csv",
                              usecols=['o_lng', 'o_lat', 'd_lng', 'd_lat', 'o', 'd'])
    o_d_lng_lat['od'] = o_d_lng_lat['o'] + o_d_lng_lat['d']
    # Eliminar filas duplicadas según od
    o_d_lng_lat = o_d_lng_lat.drop_duplicates(subset=['od'], keep='first', inplace=False)
    del o_d_lng_lat['od']
    del o_d_lng_lat['o']
    del o_d_lng_lat['d']
    # del o_d_lng_lat['pid_ratio']
    print("o_d_lng_lat =", o_d_lng_lat[:od_num])
    # Se convierte en un arreglo bidimensional
    o_d_lng_lat_feature = o_d_lng_lat.values
    # El onehot encoding puede mejorar el efecto 
    o_d_lng_lat_feature = enc.fit_transform(o_d_lng_lat_feature)
    o_d_lng_lat_feature = o_d_lng_lat_feature[:od_num]
    # Atributos de o
    o_lng_lat = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_nonoise.csv",
                              usecols=['o_lng', 'o_lat', 'o', 'd'])
    o_lng_lat['od'] = o_lng_lat['o'] + o_lng_lat['d']

    # No necesariamente se usan todos los nodos
    o_lng_lat = o_lng_lat[:num]
    print("o_lng_lat =", o_lng_lat)
    # Eliminar duplicados según o
    o_lng_lat = o_lng_lat.drop_duplicates(subset=['o'], keep='first', inplace=False)
    del o_lng_lat['o']
    del o_lng_lat['d']
    del o_lng_lat['od']

    # Se convierte en un arreglo bidimensional
    o_lng_lat_feature = o_lng_lat.values
    # El onehot encoding mejora el efecto
    o_lng_lat_feature = enc.fit_transform(o_lng_lat_feature)

    # Atributos de d
    d_lng_lat = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_nonoise.csv",
                              usecols=['d_lng', 'd_lat', 'o', 'd'])
    d_lng_lat['od'] = d_lng_lat['o'] + d_lng_lat['d']
    d_lng_lat = d_lng_lat[:num]
    # Eliminar duplicados según d
    d_lng_lat = d_lng_lat.drop_duplicates(subset=['d'], keep='first', inplace=False)
    del d_lng_lat['o']
    del d_lng_lat['d']
    del d_lng_lat['od']

    # Se convierte en un arreglo bidimensional
    d_lng_lat_feature = d_lng_lat.values
    # El onehot encoding puede mejorar el efecto
    d_lng_lat_feature = enc.fit_transform(d_lng_lat_feature)
    # Equivale a extraer las características del modo de transporte
    features = torch.FloatTensor(termfeature[:num])
    print("features =", features, features.shape)
    print("len(features) =", len(features))
    pid_feature = torch.FloatTensor(origin_feature_new)
    od_feature = torch.FloatTensor(o_d_lng_lat_feature)
    o_feature = torch.FloatTensor(o_lng_lat_feature)
    d_feature = torch.FloatTensor(d_lng_lat_feature)
    # No es conveniente reducir dimensiones aquí, los parámetros no pueden aprenderse y producirá error
    # Tratar de reducir od_feature a la dimensión de o_feature + d_feature
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)
    # Asignar etiquetas
    # Número de clases
    num_classes = 11
    # ¿Máscara? -> Para separar entrenamiento, validación y test
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))

    train_idx = np.where(float_mask <= 0.7)[0]
    val_idx = np.where((float_mask > 0.7) & (float_mask <= 0.8))[0]
    test_idx = np.where(float_mask > 0.8)[0]

    # Aquí está claro: se basa principalmente en SID
    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    # Dividir el conjunto de prueba en 4 partes
    a = np.array_split(test_idx, 4)
    test_idx_0 = a[0]
    test_idx_1 = a[1]
    test_idx_2 = a[2]
    test_idx_3 = a[3]
    test_mask_0 = get_binary_mask(num_nodes, test_idx_0)
    test_mask_1 = get_binary_mask(num_nodes, test_idx_1)
    test_mask_2 = get_binary_mask(num_nodes, test_idx_2)
    test_mask_3 = get_binary_mask(num_nodes, test_idx_3)
    # ----->Se podría considerar añadir frecuencia como peso -> ¿Mejoraría? Falta verificar
    o_d_count = pd.read_csv(path + who + '/o_d_count.csv')
    del o_d_count['Unnamed: 0']
    o_d_count = o_d_count[:num]
    # pd.DataFrame to numpy
    o_d_count = o_d_count.values
    # Se aplana de 2D a 1D
    o_d_count = o_d_count.reshape(-1, 1)
    o_list, d_list = extract_od_list()
    o_list = o_list[:num]
    d_list = d_list[:num]
    # o->d
    o_d_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (o_list, d_list)})
    # h = o_d_g.dstdata['ft']

    # d->o
    d_o_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (d_list, o_list)})
    
    return hg, o_d_g, d_o_g, features, pid_feature, o_feature, d_feature, od_feature, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask, test_mask_0, test_mask_1, test_mask_2, test_mask_3, torch.FloatTensor(o_d_count)


def load_data(dataset, remove_self_loop=False):
    if dataset == "Mydataset":
        return load_mydata_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        # -->Si se entrena simultáneamente, habría que habilitarlo para entrenar por turnos
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        # self.filename = 'early_stop.pth'
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            # Cargar el último modelo y continuar entrenando
            # self.load_checkpoint(model)
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
