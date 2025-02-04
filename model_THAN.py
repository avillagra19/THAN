import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv
from dgl.nn.pytorch import HeteroGraphConv
import pandas as pd
import dgl.function as fn
# change your computer’s path
path = '../p38dglproject/dataset/output/'
who = 'beijing'
hid_dim = 8
# Transformación lineal -> reducción de dimensiones
# Entrada: el tensor que se desea reducir y la dimensión deseada de salida
# Salida: retorna el tensor con la nueva dimensión
def LinearChange(res, dst_feat_D):
    fc = nn.Linear(res.shape[1], dst_feat_D, bias=True)
    Ba = nn.BatchNorm1d(dst_feat_D)
    Dr = nn.Dropout(0.2)

    dst_feat = fc(res)
    dst_feat = Ba(dst_feat)
    dst_feat = F.leaky_relu(dst_feat)
    dst_feat = Dr(dst_feat)
    return dst_feat

# -->Atención a nivel semántico
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        # -->El mapeo se parece a un MLP de una sola capa, para calcular el peso w
        self.project = nn.Sequential(
            # z
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # q
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # -->z proviene de las incrustaciones semánticas (semantic_embeddings)
        # -->Según la fórmula (7), se calcula el peso de cada meta-path
        w = self.project(z).mean(0)                    # (M, 1)
        # -->Operación de normalización (fórmula (8))
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        # -->Fórmula (9) a nivel semántico
        return (beta * z).sum(1)                       # (N, D * K)

# Atención a nivel de tipos de nodo heterogéneo
class NodetypeAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(NodetypeAttention, self).__init__()
        # -->El mapeo se parece a un MLP de una sola capa, para calcular el peso w
        self.project = nn.Sequential(
            # z
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # q
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # -->z proviene de las incrustaciones semánticas (semantic_embeddings)
        # -->Según la fórmula (7), se calcula el peso de cada meta-path
        w = self.project(z).mean(0)                    # (M, 1)
        # -->Operación de normalización (fórmula (8))
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        # -->Fórmula (9) a nivel semántico
        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        # --> ¿Atención a nivel de nodos?
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            # HAN usa GAT para la operación basada en meta-path
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, residual=False, activation=F.elu,
                                           allow_zero_in_degree=True))

        # --> Atención a nivel semántico
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}


    def forward(self, g, h):
        # -->Incrustación semántica -> atención a nivel de nodo
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        # -->Incrustación semántica
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        # --> Incrustación semántica: semantic_embeddings se usa como entrada para la atención semántica
        # -->Atención a nivel semántico
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class THANLayer(nn.Module):
    def __init__(self, in_size, pid_size, od_size, out_size, layer_num_heads, dropout):
        super(THANLayer, self).__init__()
        # pid_gat
        # Grafo bipartito (nodo origen, nodo destino), salida de características
        self.pid_gat = GATConv((pid_size, in_size), out_size, layer_num_heads,
                               dropout, dropout, residual=True, activation=F.leaky_relu,
                               allow_zero_in_degree=True)
        # gat
        self.od_gat = GATConv((od_size, in_size), out_size, layer_num_heads,
                              dropout, dropout, residual=True, activation=F.leaky_relu,
                              allow_zero_in_degree=True)

        # --> ¿Atención a nivel semántico?
        self.nodetype_attention = NodetypeAttention(in_size=out_size * layer_num_heads)

    def forward(self, temp_h, pid_g, pid_h, od_g, od_h):
        # -->Incrustación de tipo de nodo -> atención a nivel de nodo
        nodetype_embeddings = []
        nodetype_embeddings.append(self.pid_gat(pid_g, (pid_h, temp_h)).flatten(1))
        nodetype_embeddings.append(self.od_gat(od_g, (od_h, temp_h)).flatten(1))
        # --> Incrustación semántica
        nodetype_embeddings = torch.stack(nodetype_embeddings, dim=1)                  # (N, M, D * K)
        # --> Incrustación semántica: semantic_embeddings como entrada a la atención semántica
        # -->Atención a nivel semántico
        return self.nodetype_attention(nodetype_embeddings)
# --Atención a nivel de tipos de nodo heterogéneo
class HybridAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(HybridAttention, self).__init__()
        # -->El mapeo se parece a un MLP de una sola capa, para calcular el peso w
        self.project = nn.Sequential(
            # z
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # q
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class THAN(nn.Module):
    def __init__(self, meta_paths, in_size, pid_size, o_size, d_size, od_size, hidden_size, out_size, num_heads, dropout):
        super(THAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        # Mecanismo de atención multi-cabezal (fórmula (5)); num_heads indica el número de cabezales
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))

        # -->Cantidad de datos x dimensión 3075
        self.predict0 = nn.Linear(hidden_size * num_heads[-1], out_size, bias=True)
        self.Ba0 = nn.BatchNorm1d(out_size)
        self.Dr0 = nn.Dropout(0.2)

        # Atención de nodo heterogéneo
        # hidden_size = 8, num_heads[0] = num_heads[-1] = 8
        self.nodetype_nn = THANLayer(in_size, pid_size, od_size, hidden_size, num_heads[0], dropout)
        # MLP para predicción. Se podría usar múltiples capas, aunque parece que no es tan bueno

        self.predict1 = nn.Linear(hidden_size * num_heads[-1] * 2, out_size, bias=True)
        self.Ba1 = nn.BatchNorm1d(out_size)
        self.Dr1 = nn.Dropout(0.2)

        # Vectores heterogéneos, vectores homogéneos
        self.Hybrid_attention = HybridAttention(in_size=hidden_size * num_heads[-1])

        self.predict2 = nn.Linear(in_size, out_size, bias=True)

        # Grafo bipartito (nodo origen, nodo destino), salida de características
        # La dimensión out_size no puede ser la misma que la dimensión original. De lo contrario, todo bien
        self.o_d_gat = GraphConv(o_size, o_size, norm='both', weight=True, bias=True)#, activation=torch.tanh)

        # Origen->Destino con GCN
        self.d_o_gat = GraphConv(d_size, d_size, norm='both', weight=True, bias=True)#, activation=torch.tanh)

        # od_pid_gat
        # Origen->Destino con GCN
        self.od_pid_gat = GATConv((od_size, pid_size), od_size, num_heads[0],
                                           dropout, dropout, residual=True, activation=F.leaky_relu,
                                           allow_zero_in_degree=True)

        # pid_od_gat
        self.pid_od_gat = GATConv((pid_size, od_size), pid_size, num_heads[0],
                               dropout, dropout, residual=True, activation=F.leaky_relu,
                               allow_zero_in_degree=True)

        # Recuperar dimensión de o
        self.recover_o_D = nn.Linear(d_size, o_size, bias=True)
        # Recuperar dimensión de d
        self.recover_d_D = nn.Linear(o_size, d_size, bias=True)
        # Recuperar dimensión de pid
        self.recover_pid_D = nn.Linear(od_size, pid_size, bias=True)
        # Recuperar dimensión de od
        self.recover_od_D = nn.Linear(pid_size, od_size, bias=True)

        # Convertir la dimensión de (o+d) a la dimensión od
        self.recover_o_d_to_od_D = nn.Linear((o_size + d_size), od_size, bias=True)
        hidden = 128
        # Recuperar dimensión de pid -> in_size
        self.recover_pid_D_hetergcn = nn.Linear(pid_size, in_size, bias=True)
        # Recuperar dimensión de od -> in_size
        self.recover_od_D_hetergcn = nn.Linear(od_size, in_size, bias=True)
        # https://docs.dgl.ai/en/0.6.x/guide/nn-heterograph.html
        self.heterGcn = HeteroGraphConv({
            # p es el nodo origen, a el nodo destino
            'pa': GraphConv(in_size, hidden),
            'ap': GraphConv(in_size, hidden),
            'pf': GraphConv(in_size, hidden),
            'fp': GraphConv(in_size, hidden)},
            aggregate='sum')
        self.heterGcn1 = HeteroGraphConv({
            # p es el nodo origen, a el nodo destino
            'pa': GraphConv(hidden, hidden_size * num_heads[-1]),
            'ap': GraphConv(hidden, hidden_size * num_heads[-1]),
            'pf': GraphConv(hidden, hidden_size * num_heads[-1]),
            'fp': GraphConv(hidden, hidden_size * num_heads[-1])},
            aggregate='sum')
        self.Dr3 = nn.Dropout(0.6)

    # pid_h son las características de pid
    # od_h son las características de od
    # THAN completo: o + d -> od, od y pid se influyen mutuamente, mecanismo de atención por capas
    # THAN: incrustación de múltiples grafos bipartitos, nodos heterogéneos con GAT, meta-path (nodos homogéneos) con GAT, con residual

    #  --->
    #  (1) Validación por grupos para comprobar robustez
    #  (2) Sensibilidad de hiperparámetros: variación de dimensión en vectores homogéneos y heterogéneos, número de cabezales de atención
    # forward_TAHN
    def forward(self, g, h, pid_h, o_h, d_h, o_d_g, d_o_g, od_h, o_d_od_ID_data, o_d_count):
        # Se hace una copia de seguridad de las características originales del nodo
        temp_h = h
        # Predicción de nodos en red homogénea -> basada en meta-path
        for gnn in self.layers:
            # Cada vez se usa el h(original). Se cumple h(actualizado) = L*h(original)*w + b
            # En realidad, se está cambiando w para que h(actualizado) se ajuste mejor a la predicción de la etiqueta
            h = gnn(g, h)

        # ---->Arriba se hace la predicción m basada en meta-path. Abajo, la predicción m basada en nodos heterogéneos

        # -------->Combinar o + d -> od, uniendo o y d en od<--------
        # Actualizar características del nodo o
        res0 = self.o_d_gat(o_d_g, (o_h, d_h))  # , edge_weight=o_d_count)
        # Operación de reducción de dimensión -> con GCN no es necesaria -> no hay mecanismo multi-cabezal
        # Actualizar características del nodo d
        res1 = self.d_o_gat(d_o_g, (d_h, o_h))  # , edge_weight=o_d_count)
        # Operación de reducción de dimensión -> con GCN no es necesaria -> no hay mecanismo multi-cabezal

        # Transformación lineal para recuperar la dimensión de o_h y d_h
        o_h = self.recover_o_D(res1)
        d_h = self.recover_d_D(res0)
        o_h = o_h.detach().numpy()
        d_h = d_h.detach().numpy()
        o_h = pd.DataFrame(o_h)
        d_h = pd.DataFrame(d_h)
        o_df = o_h.reset_index()
        d_df = d_h.reset_index()
        o_h['o_ID'] = range(len(o_df))
        d_h['d_ID'] = range(len(d_df))
        # La lectura en bucle supone un coste fijo, quizás se pueda sacar fuera
        # o_d_od_ID_data = pd.read_csv((path + who + '/o_d_od_ID_data.csv'))
        o_d_od_ID_data = o_d_od_ID_data.detach().numpy()
        o_d_od_ID_data = pd.DataFrame(o_d_od_ID_data)
        o_d_od_ID_data_temp = pd.DataFrame()
        o_d_od_ID_data_temp['od_ID'] = o_d_od_ID_data[1]
        o_d_od_ID_data_temp['o_ID'] = o_d_od_ID_data[2]
        o_d_od_ID_data_temp['d_ID'] = o_d_od_ID_data[3]
        # No leerlo desde archivo acelera el proceso
        o_d_od_ID_data = o_d_od_ID_data_temp
        o_d_od_ID_data = o_d_od_ID_data.merge(o_h, on='o_ID', how='left')
        o_d_od_ID_data = o_d_od_ID_data.merge(d_h, on='d_ID', how='left')
        # Eliminar filas duplicadas según pid
        o_d_od_ID_data = o_d_od_ID_data.drop_duplicates(subset=['od_ID'], keep='first', inplace=False)
        # print("od number =", g.num_nodes('field'))
        # -->Se necesita el número de OD
        o_d_od_ID_data = o_d_od_ID_data[: g.num_nodes('field')]
        del o_d_od_ID_data['od_ID']
        del o_d_od_ID_data['o_ID']
        del o_d_od_ID_data['d_ID']

        # Fusionar características de o & d
        od_h = o_d_od_ID_data.values
        od_h = torch.FloatTensor(od_h)
        # Convertir las características generadas de o y d a la dimensión od
        od_h = self.recover_o_d_to_od_D(od_h)
        #  -----> o + d -->od   <--------
        # Secuencia de nodos bipartitos
        pid_m = g.edges('all', etype='pa')
        od_m = g.edges('all', etype='pf')
        # Construir grafo bipartito od_pid para aprender características de pid
        od_pid_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (od_m[1], pid_m[1])})
        # Construir grafo bipartito pid_od para aprender características de od
        pid_od_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (pid_m[1], od_m[1])})

        # Actualizar características del nodo pid: od -> pid
        res0 = self.od_pid_gat(od_pid_g, (od_h, pid_h))
        # Con GCN no hace falta reducción de dimensión
        res0 = res0.mean(axis=1, keepdim=False)  # Se hace la media y se convierte a bidimensional
        # Actualizar características del nodo od: pid -> od
        res1 = self.pid_od_gat(pid_od_g, (pid_h, od_h))
        # Con GCN no hace falta reducción de dimensión
        res1 = res1.mean(axis=1, keepdim=False)  # Se hace la media y se convierte a bidimensional

        # Transformación lineal para recuperar las dimensiones de pid_h y od_h
        pid_h = self.recover_pid_D(res0)
        od_h = self.recover_od_D(res1)

        # Construir grafo bipartito -> volver a usar la atención en grafo para aprender características de mode
        pid_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (pid_m[1], pid_m[2])})
        # Construir grafo bipartito -> volver a usar la atención en grafo para aprender características de nodos
        od_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (od_m[1], od_m[2])})
        # Predicción de nodos en red heterogénea -> basada en nodos heterogéneos
        m_h = self.nodetype_nn(temp_h, pid_g, pid_h, od_g, od_h)

        # Formato de fusión de las características del nodo objetivo -> concatenación simple
        h = torch.cat((m_h, h), 1)
        # Usar un MLP. Se conectan ambas informaciones entrenadas
        h = self.predict1(h)  # Cantidad de datos
        # THAN realiza esta operación (función de activación, normalización, etc.)
        h = self.Ba1(h)
        h = F.leaky_relu(h)
        # h = torch.tanh(h)
        h = F.softmax(h, dim=1)
        h = self.Dr1(h)

        return h
